"""Deterministic editable DOCX tables, using Python's standard library only.

Run this standalone renderer with the configured document-authoring interpreter.
It consumes aggregate table JSON and the actual Arial font files; it does not
load study inputs, fit models, or require changes to the scientific environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from uuid import UUID
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'


def _font(data, bold):
    """Check TrueType family/style and embedding permissions from actual bytes."""
    if data[:4] not in (b'\x00\x01\x00\x00',b'OTTO'):
        raise ValueError('Configured Arial font must be a TrueType or OpenType file.')
    count = struct.unpack_from('>H',data,4)[0]
    tables = {}
    for i in range(count):
        tag,checksum,offset,length = struct.unpack_from('>4sIII',data,12+16*i)
        if offset+length > len(data):
            raise ValueError('Invalid font table bounds.')
        tables[tag] = (offset,length)
    offset,length = tables[b'OS/2']
    flags = struct.unpack_from('>H',data,offset+8)[0]
    if flags & (0x0002|0x0200):
        raise ValueError('The configured font does not permit this embedding.')
    start,length = tables[b'name']
    _,count,strings = struct.unpack_from('>HHH',data,start)
    names = {}
    for i in range(count):
        platform,encoding,language,name_id,size,offset = struct.unpack_from('>HHHHHH',data,start+6+12*i)
        if platform in (0,3):
            value = data[start+strings+offset:start+strings+offset+size].decode('utf-16-be')
            names.setdefault(name_id,set()).add(value)
    family = names.get(1,set()) | names.get(16,set())
    style = ' '.join(names.get(2,set()) | names.get(17,set())).lower()
    if 'Arial' not in family or ('bold' in style) != bold or 'italic' in style or 'oblique' in style:
        raise ValueError('Document fonts must be actual Arial regular and Arial Bold.')
    return hashlib.sha256(data).hexdigest()


def _p(text, *, bold=False, size=20, align='left', keep=False, style=None, after=40):
    pp = (f'<w:pStyle w:val="{style}"/>' if style else '')
    pp += f'<w:spacing w:after="{after}" w:line="240" w:lineRule="auto"/><w:jc w:val="{align}"/>'
    if keep:
        pp += '<w:keepNext/>'
    runs = []
    for i,line in enumerate(text.split('\n')):
        if i:
            runs.append('<w:r><w:br/></w:r>')
        # Arial does not contain every Unicode subscript numeral. Native Word
        # subscript runs use actual Arial digit glyphs, avoiding silent fallback.
        for chunk in re.split('([₀₁₂₃₄₅₆₇₈₉]+)',line):
            if not chunk:continue
            subscript=chunk[0] in '₀₁₂₃₄₅₆₇₈₉'
            if subscript:chunk=chunk.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789'))
            runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>'
                        f'<w:sz w:val="{size}"/><w:color w:val="000000"/>'+('<w:b/>' if bold else '')+
                        ('<w:vertAlign w:val="subscript"/>' if subscript else '')+
                        f'</w:rPr><w:t xml:space="preserve">{escape(chunk)}</w:t></w:r>')
    return '<w:p><w:pPr>'+pp+'</w:pPr>'+''.join(runs)+'</w:p>'


def _table(table):
    n = len(table['columns'])
    proportions = {
        'main_table_1':[.44,.186,.187,.187], 'main_table_2':[.20,.14,.33,.33],
        'appendix_table_s1':[.31,.21,.135,.21,.135],
        'appendix_table_s2':[.26,.185,.185,.185,.185],
        'appendix_table_s3':[.34,.14,.26,.26]}.get(table['id'],[.18,.36,.46] if n==3 else [.24]+[.152]*5)
    if len(proportions)!=n:
        raise ValueError('Unknown manuscript table width specification.')
    widths = [round(10440*v) for v in proportions]
    widths[-1] += 10440-sum(widths)
    borders = ''.join(f'<w:{side} w:val="single" w:sz="4" w:color="D9D9D9"/>'
                      for side in ('top','left','bottom','right','insideH','insideV'))
    pr = ('<w:tblPr><w:tblW w:w="10440" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
          '<w:tblBorders>'+borders+'</w:tblBorders><w:tblCellMar>'+
          ''.join(f'<w:{s} w:w="{35 if table["id"]=="main_table_1" and s in ("top","bottom") else 75}" w:type="dxa"/>' for s in ('top','left','bottom','right'))+
          '</w:tblCellMar></w:tblPr>')
    grid = '<w:tblGrid>'+''.join(f'<w:gridCol w:w="{v}"/>' for v in widths)+'</w:tblGrid>'
    rows = []
    for row in table['rows']:
        if row['kind']=='spacer':
            continue
        header = row['kind']=='header'
        heading = row['kind'] in {'section','category_heading'}
        cells, column = [],0
        for item in row['cells']:
            span = item.get('colspan',1)
            if type(span) is not int or span<1 or column+span>n:
                raise ValueError('Invalid manuscript table span.')
            width = sum(widths[column:column+span])
            cp = f'<w:tcW w:w="{width}" w:type="dxa"/><w:vAlign w:val="center"/>'
            if span>1:
                cp += f'<w:gridSpan w:val="{span}"/>'
            if header or heading:
                cp += '<w:shd w:val="clear" w:fill="'+('E8EEF3' if header else 'F3F3F3')+'"/>'
            align = 'left' if column==0 else 'center'
            size = 19 if n>=5 or table['id']=='main_table_1' else 20
            cells.append('<w:tc><w:tcPr>'+cp+'</w:tcPr>'+_p(item['text'],bold=header or heading,
                         size=size,align=align,keep=heading,after=0)+'</w:tc>')
            column += span
        if column!=n:
            raise ValueError('Manuscript row does not fill its declared grid.')
        rp = '<w:cantSplit/>'+('<w:tblHeader/>' if header else '')
        rows.append('<w:tr><w:trPr>'+rp+'</w:trPr>'+''.join(cells)+'</w:tr>')
    return '<w:tbl>'+pr+grid+''.join(rows)+'</w:tbl>'


def document_bytes(tables, arial, arial_bold):
    import io
    fonts = [Path(arial).read_bytes(),Path(arial_bold).read_bytes()]
    hashes = [_font(data,bool(i)) for i,data in enumerate(fonts)]
    content = [_p('DCHS lung function manuscript tables',style='Title',size=30,bold=True,after=100),
               _p('Editable tables generated from verified aggregate results, rounded only for display.',size=20,after=120)]
    for i,table in enumerate(tables):
        if i:
            content.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        content.append(_p(table['title'],bold=True,size=22,keep=True,after=120))
        content.append(_table(table))
        content.append(_p('',size=8,after=0))
        for note in table['footnotes']:
            content.append(_p(note,size=19,after=90))
    section = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="360" w:footer="360"/></w:sectPr>'
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>'+''.join(content)+section+'</w:body></w:document>'
    styles = f'<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="20"/><w:color w:val="000000"/></w:rPr></w:rPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/><w:color w:val="000000"/></w:rPr></w:style></w:styles>'
    guids = [UUID('7B32AF7C-124E-43DC-9731-485348FC0201'),UUID('7B32AF7C-124E-43DC-9731-485348FC0202')]
    fonttable = f'<w:fonts xmlns:w="{W}" xmlns:r="{R}"><w:font w:name="Arial"><w:family w:val="swiss"/>'
    for i,kind in enumerate(('Regular','Bold')):
        fonttable += f'<w:embed{kind} r:id="font{i}" w:fontKey="{{{str(guids[i]).upper()}}}"/>'
    fonttable += '</w:font></w:fonts>'
    fontrels = f'<Relationships xmlns="{PKG}">'+''.join(f'<Relationship Id="font{i}" Type="{R}/font" Target="fonts/arial{i}.odttf"/>' for i in range(2))+'</Relationships>'
    rels = f'<Relationships xmlns="{PKG}"><Relationship Id="document" Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>'
    docrels = f'<Relationships xmlns="{PKG}"><Relationship Id="styles" Type="{R}/styles" Target="styles.xml"/><Relationship Id="fonts" Type="{R}/fontTable" Target="fontTable.xml"/></Relationships>'
    types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="odttf" ContentType="application/vnd.openxmlformats-officedocument.obfuscatedFont"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/></Types>'
    entries = {'[Content_Types].xml':types,'_rels/.rels':rels,'word/document.xml':document,
               'word/styles.xml':styles,'word/fontTable.xml':fonttable,
               'word/_rels/document.xml.rels':docrels,'word/_rels/fontTable.xml.rels':fontrels}
    for i,data in enumerate(fonts):
        data = bytearray(data)
        key = guids[i].bytes[::-1]
        for j in range(32):
            data[j] ^= key[j%16]
        entries[f'word/fonts/arial{i}.odttf'] = bytes(data)
    buffer = io.BytesIO()
    with ZipFile(buffer,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(entries):
            info = ZipInfo(name,(2000,1,1,0,0,0));info.compress_type=ZIP_DEFLATED
            info.create_system=3;info.external_attr=0o600<<16
            data=entries[name]
            z.writestr(info,data.encode('utf-8') if isinstance(data,str) else data,compresslevel=9)
    return buffer.getvalue(), hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--arial',required=True,type=Path)
    parser.add_argument('--arial-bold',required=True,type=Path)
    args=parser.parse_args()
    tables=json.loads(args.tables.read_text())['tables']
    data,_=document_bytes(tables,args.arial,args.arial_bold)
    if args.output.exists():
        raise ValueError('Document output already exists.')
    args.output.write_bytes(data)


if __name__=='__main__':
    main()
