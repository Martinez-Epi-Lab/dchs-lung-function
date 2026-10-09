"""Publication figures drawn only from the resolved, keyed aggregate payload.

This module controls typography and layout. It neither selects analytical rows
nor calculates estimates, intervals, sample counts, or manuscript annotations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


FIGURE_IDS = (
    'main_figure_1', 'main_figure_2', 'main_figure_3',
    'appendix_figure_s1', 'appendix_figure_s2', 'appendix_figure_s3',
    'appendix_figure_s4', 'appendix_figure_s5', 'appendix_figure_s6',
)
INK = '#202A33'
MUTED = '#586570'
GRID = '#E4E8EB'
MARKERS = {'circle': 'o', 'square': 's', 'diamond': 'D', 'triangle': '^'}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value):
    """Use native Arial digits for subscripts, with no fallback glyphs."""
    value = str(value).translate(str.maketrans({'−': '-', '–': '-', '—': '-', '‑': '-', '\u00a0': ' '}))
    return re.sub('[₀₁₂₃₄₅₆₇₈₉]+', lambda m: '$_{' + m.group().translate(
        str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')) + '}$', value)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _drawable(row):
    values = [row[k] for k in ('point', 'lower', 'upper')]
    if not all(_finite(v) for v in values):
        if any(v is not None for v in values):
            raise ValueError('A plotted interval must be complete and finite or explicitly unavailable.')
        return False
    if values[1] > values[2]:
        raise ValueError('A plotted interval must have ordered bounds.')
    return True


def _limits(rows, axis):
    values = [v for r in rows if _drawable(r) for v in (r['point'], r['lower'], r['upper'])]
    values.extend(axis.get('ticks', []))
    if axis.get('null') is not None:
        values.append(axis['null'])
    if not values:
        raise ValueError('No finite axis range can be derived for this panel.')
    logarithmic = axis.get('scale', 'linear') == 'log'
    if logarithmic and min(values) <= 0:
        raise ValueError('A logarithmic panel requires positive estimates and interval limits.')
    if 'limits' in axis:
        lo, hi = axis['limits']
        if not lo <= min(values) <= max(values) <= hi or lo >= hi:
            raise ValueError('Configured limits would clip a plotted interval or tick.')
        return [float(lo), float(hi)]
    values = [math.log(v) for v in values] if logarithmic else values
    low, high = min(values), max(values)
    span = high - low or max(abs(low) * .1, .1)
    low, high = low - .08 * span, high + .08 * span
    return [math.exp(low), math.exp(high)] if logarithmic else [low, high]


class _Canvas:
    def __init__(self, width, height, fonts):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties
        self.fig = Figure(figsize=(width, height), dpi=120, facecolor='white')
        self.canvas = FigureCanvasAgg(self.fig)
        self.width, self.height = width, height
        self.fonts = fonts
        self.regular = FontProperties(fname=str(fonts['arial']))
        self.bold = FontProperties(fname=str(fonts['arial_bold']))
        self.text_items = []
        self.panel_metadata = []

    def font(self, size, bold=False):
        prop = (self.bold if bold else self.regular).copy()
        prop.set_size(size)
        return prop

    def wrap(self, value, width, size=10, bold=False):
        """Wrap whole words using the actual font's rendered advance widths."""
        renderer = self.canvas.get_renderer()
        result = []
        for paragraph in str(value).split('\n'):
            line = ''
            for word in paragraph.split():
                proposed = word if not line else line + ' ' + word
                prepared = _text(proposed)
                pixels = renderer.get_text_width_height_descent(prepared, self.font(size, bold),
                                                               ismath='$' in prepared)[0]
                if line and pixels > width * self.fig.dpi:
                    result.append(line)
                    line = word
                else:
                    line = proposed
            result.append(line)
        return '\n'.join(result)

    def label(self, x, y, value, *, size=10, bold=False, color=INK,
              ha='left', va='top', width=None, rotation=0):
        if width is not None:
            value = self.wrap(value, width, size, bold)
        artist = self.fig.text(x / self.width, y / self.height, _text(value),
                               fontproperties=self.font(size, bold), color=color,
                               ha=ha, va=va, rotation=rotation, linespacing=1.22)
        self.text_items.append(artist)
        return artist

    def axes(self, left, bottom, width, height):
        return self.fig.add_axes([left / self.width, bottom / self.height,
                                 width / self.width, height / self.height])

    def line(self, x1, y1, x2, y2, color=GRID, linewidth=.7, linestyle='-'):
        from matplotlib.lines import Line2D
        self.fig.add_artist(Line2D([x1 / self.width, x2 / self.width],
            [y1 / self.height, y2 / self.height], transform=self.fig.transFigure,
            color=color, linewidth=linewidth, linestyle=linestyle))

    def audit(self):
        from matplotlib.text import Text
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        box = self.fig.bbox
        outside = []
        fonts = set()
        for artist in self.fig.findobj(match=Text):
            if not artist.get_visible() or not artist.get_text():
                continue
            bbox = artist.get_window_extent(renderer)
            # Tick labels not selected for display can remain as invisible
            # children; only visible labels with a nonempty extent are checked.
            if bbox.width and bbox.height and (bbox.x0 < box.x0 - 1 or bbox.y0 < box.y0 - 1
                    or bbox.x1 > box.x1 + 1 or bbox.y1 > box.y1 + 1):
                outside.append(artist.get_text())
            fonts.add(artist.get_fontproperties().get_name())
        if outside:
            raise ValueError('Figure text extends outside the page: ' + repr(outside))
        if fonts != {'Arial'}:
            raise ValueError('Every figure text object must use actual Arial: ' + repr(fonts))
        for axis in self.fig.axes:
            xlabel = axis.xaxis.label
            if xlabel.get_visible() and xlabel.get_text():
                label_box = xlabel.get_window_extent(renderer)
                if any(label.get_visible() and label.get_text() and
                       label_box.overlaps(label.get_window_extent(renderer))
                       for label in axis.get_xticklabels()):
                    raise ValueError('An x-axis title overlaps its tick labels.')
        return {'text_within_page': True, 'text_font_families': sorted(fonts),
                'x_axis_titles_clear_of_ticks': True,
                'text_objects_checked': sum(bool(t.get_text()) for t in self.fig.findobj(match=Text))}


def _style_axis(ax, canvas, axis, *, orientation='horizontal'):
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
    setter = ax.set_xscale if orientation == 'horizontal' else ax.set_yscale
    setter(axis.get('scale', 'linear'))
    variable = ax.xaxis if orientation == 'horizontal' else ax.yaxis
    if 'ticks' in axis:
        (ax.set_xticks if orientation == 'horizontal' else ax.set_yticks)(axis['ticks'])
    elif axis.get('scale', 'linear') == 'linear':
        variable.set_major_locator(MaxNLocator(nbins=5))
    variable.set_minor_locator(NullLocator())
    variable.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.tick_params(axis='both', labelsize=9, colors=MUTED, width=.65, length=3.5)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(canvas.font(9))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_linewidth(.7)
    ax.spines['left'].set_linewidth(.7)
    ax.set_axisbelow(True)
    if axis.get('null') is not None:
        (ax.axvline if orientation == 'horizontal' else ax.axhline)(
            axis['null'], color='#929CA4', linewidth=1, linestyle=(0, (3, 3)), zorder=1)


def _series_style(panel, series):
    style = panel['series_styles'][series]
    return {'color': style['color'], 'marker': MARKERS.get(style.get('marker', 'circle'), 'o'),
            'markerfacecolor': 'white' if style.get('fill') == 'open' else style['color'],
            'markeredgecolor': style['color'], 'markersize': 6, 'markeredgewidth': 1.2}


def _legend(canvas, panels, y, *, width, left=.5):
    from matplotlib.lines import Line2D
    styles = {}
    for panel in panels:
        for series in panel.get('series_order', []):
            style = panel['series_styles'][series]
            styles.setdefault(style['label'], (panel, series))
    if len(styles) < 2:
        return 0
    labels = list(styles)
    multiline = any(len(label) > 48 for label in labels) or sum(len(label) for label in labels) > 100
    rows = len(labels) if multiline else 1
    for i, label in enumerate(labels):
        panel, series = styles[label]
        x = left if multiline else left + i * width / len(labels)
        yy = y - (i * .24 if multiline else 0)
        style = _series_style(panel, series)
        canvas.fig.add_artist(Line2D([x / canvas.width, (x + .25) / canvas.width],
            [yy / canvas.height] * 2, transform=canvas.fig.transFigure, linewidth=1.5, **style))
        canvas.label(x + .35, yy, label, size=9, va='center', width=width - .4 if multiline else width / len(labels) - .4)
    return .24 * rows


def _forest_horizontal(canvas, panel, box, *, compact=False):
    left, bottom, width, height = box
    rows = panel['rows']
    if [r['id'] for r in rows] != panel['row_order']:
        raise ValueError('Forest row order differs from the supplied publication order.')
    support = bool(panel.get('support_annotation'))
    label_width = width * (.36 if support else .27)
    annotation_width = width * (.235 if support else .26)
    support_width = width * .11 if support else 0
    gap = .13
    plot_left = left + label_width + gap
    plot_width = width - label_width - annotation_width - support_width - 3 * gap
    annotation_left = plot_left + plot_width + gap + support_width
    title_y = bottom + height
    canvas.label(left, title_y, panel['id'] + '  ' + panel['title'], size=11 if compact else 12,
                 bold=True, width=width)
    title_lines = len(canvas.wrap(panel['id'] + '  ' + panel['title'], width, 11 if compact else 12, True).splitlines())
    top = title_y - .25 * title_lines - .46
    plot_bottom = bottom + .60
    plot_height = top - plot_bottom
    if plot_height <= .5:
        raise ValueError('The forest panel is too short for its labels and axes.')
    positions, pos, previous = [], 0, None
    separated_series = support
    for row in rows:
        if separated_series and previous is not None and row['series'] != previous:
            pos += .60
        positions.append(pos)
        pos += 1
        previous = row['series']
    ylim = [positions[-1] + .7, -.7]
    ax = canvas.axes(plot_left, plot_bottom, plot_width, plot_height)
    ax.set_ylim(ylim)
    ax.set_yticks([])
    _style_axis(ax, canvas, panel['value_axis'])
    ax.spines['left'].set_visible(False)
    limits = _limits(rows, panel['value_axis'])
    ax.set_xlim(limits)
    canvas.label(annotation_left, top + .22, panel['interval'], size=8.5, bold=True,
                 color=MUTED, width=annotation_width, va='bottom')
    if support:
        canvas.label(plot_left + plot_width + gap, top + .22,
                     panel['support_annotation']['heading'], size=8.5, bold=True, color=MUTED, va='bottom')
    last_label = None
    for i, (row, y) in enumerate(zip(rows, positions)):
        if _drawable(row):
            style = _series_style(panel, row['series'])
            # An equal-tailed interval need not contain a posterior mean.
            # Draw the supplied interval and point independently.
            ax.plot([row['lower'], row['upper']], [y, y], color=style['color'], linewidth=1.3, zorder=2)
            ax.plot([row['lower'], row['upper']], [y, y], linestyle='none', marker='|',
                    markersize=6, markeredgewidth=1, color=style['color'], zorder=2)
            ax.plot(row['point'], y, linestyle='none', zorder=3, **style)
        yy = plot_bottom + plot_height * (ylim[0] - y) / (ylim[0] - ylim[1])
        if row['label'] != last_label:
            same = [j for j in range(i, len(rows)) if rows[j]['label'] == row['label']]
            center = sum(positions[j] for j in same) / len(same)
            label_y = plot_bottom + plot_height * (ylim[0] - center) / (ylim[0] - ylim[1])
            canvas.label(left, label_y, row['label'], size=9 if compact else 10,
                         width=label_width - .04, va='center')
        last_label = row['label']
        if panel['annotation']['show'] or not _drawable(row):
            canvas.label(annotation_left, yy, row['display'], size=8.8 if compact else 9.5,
                         color=panel['series_styles'][row['series']]['color'], va='center', width=annotation_width)
        if support:
            canvas.label(plot_left + plot_width + gap, yy, row['support_display'], size=9, va='center')
    ax.set_xlabel(_text(canvas.wrap(panel['value_axis']['label'], plot_width + .1, 9.5)),
                  fontproperties=canvas.font(9.5), labelpad=7)
    canvas.panel_metadata.append({'id': panel['id'], 'kind': 'forest', 'orientation': 'horizontal',
        'axis_scale': panel['value_axis']['scale'], 'axis_limits': limits, 'rows': len(rows),
        'drawn_intervals': sum(_drawable(r) for r in rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
        'interval': panel['interval'], 'estimate': panel['estimate']})


def _forest_vertical(canvas, panel, box):
    left, bottom, width, height = box
    rows = panel['rows']
    canvas.label(left, bottom + height, panel['id'] + '  ' + panel['title'], size=12, bold=True, width=width)
    ax = canvas.axes(left + .62, bottom + .66, width - .78, height - 1.10)
    _style_axis(ax, canvas, panel['value_axis'], orientation='vertical')
    limits = _limits(rows, panel['value_axis'])
    ax.set_ylim(limits)
    ax.set_xlim(-.65, len(rows) - .35)
    ax.set_xticks(range(len(rows)), [_text(r['label']) for r in rows], fontproperties=canvas.font(9.5))
    for i, row in enumerate(rows):
        if _drawable(row):
            style = _series_style(panel, row['series'])
            ax.plot([i, i], [row['lower'], row['upper']], color=style['color'], linewidth=1.3, zorder=2)
            ax.plot([i, i], [row['lower'], row['upper']], linestyle='none', marker='_',
                    markersize=8, markeredgewidth=1, color=style['color'], zorder=2)
            ax.plot(i, row['point'], linestyle='none', zorder=3, **style)
        else:
            ax.text(i, .5, _text(row['display']), transform=ax.get_xaxis_transform(),
                    fontproperties=canvas.font(9), ha='center')
    ax.set_ylabel(_text(panel['value_axis']['label']), fontproperties=canvas.font(10), labelpad=8)
    canvas.label(left + width / 2, bottom + .1, 'Age at FeNO measurement', size=10, ha='center', va='bottom')
    canvas.panel_metadata.append({'id': panel['id'], 'kind': 'forest', 'orientation': 'vertical',
        'axis_scale': panel['value_axis']['scale'], 'axis_limits': limits, 'rows': len(rows),
        'drawn_intervals': sum(_drawable(r) for r in rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
        'interval': panel['interval'], 'estimate': panel['estimate']})


def _trajectory(canvas, panel, box):
    import numpy as np
    left, bottom, width, height = box
    canvas.label(left, bottom + height, panel['id'] + '  ' + panel['title'], size=12, bold=True, width=width)
    ax = canvas.axes(left + .57, bottom + .65, width - .74, height - 1.34)
    _style_axis(ax, canvas, panel['x_axis'])
    points = panel['points']
    ax.set_xlim(panel['x_axis']['limits'])
    yrange = _limits(points, panel['y_axis'])
    ax.set_ylim(yrange)
    ax.yaxis.grid(True, color=GRID, linewidth=.65)
    for series in panel['series_order']:
        selected = [p for p in points if p['series'] == series]
        if not selected or not all(_drawable(p) for p in selected):
            raise ValueError('Every selected trajectory requires complete finite pointwise intervals.')
        selected = sorted(selected, key=lambda p: p['x'])
        x = np.array([p['x'] for p in selected])
        if len(set(x)) != len(x):
            raise ValueError('Trajectory ages must be unique within series.')
        style = panel['series_styles'][series]
        y, lower, upper = (np.array([p[k] for p in selected]) for k in ('point', 'lower', 'upper'))
        ax.fill_between(x, lower, upper, color=style['color'], alpha=.16, linewidth=0, zorder=1)
        ax.plot(x, y, color=style['color'], linewidth=2, linestyle=style.get('line', 'solid'),
                label=style['label'], zorder=2)
    ax.set_xlabel(_text(panel['x_axis']['label']), fontproperties=canvas.font(10), labelpad=7)
    ax.set_ylabel(_text(panel['y_axis']['label']), fontproperties=canvas.font(10), labelpad=7)
    legend = ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), borderaxespad=0, ncol=2,
                       frameon=False, prop=canvas.font(8.8), handlelength=2, columnspacing=1.3)
    for label in list(ax.get_yticklabels()) + list(ax.get_xticklabels()):
        label.set_fontproperties(canvas.font(9))
    canvas.panel_metadata.append({'id': panel['id'], 'kind': 'trajectory', 'axis_scale': 'linear',
        'x_limits': list(ax.get_xlim()), 'y_limits': yrange, 'points': len(points),
        'series': panel['series_order'], 'status_counts': dict(Counter(p['status'] for p in points)),
        'interval': panel['interval'], 'estimate': panel['estimate']})


def _flow(canvas, panel, box):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as DrawPath
    left, bottom, width, height = box
    # Coordinates are layout only. Every label and count comes from its node.
    positions = {
        'antenatal_enrollment': (.47, .925, .30, .080),
        'women_with_liveborn_children': (.47, .810, .30, .075),
        'liveborn_children_enrolled': (.47, .685, .30, .105),
        'children_with_pft_records': (.47, .530, .30, .110),
        'combined_eligible_lung_function': (.47, .370, .30, .105),
        'excluded_before_child_cohort': (.145, .925, .265, .090),
        'antenatal_loss_to_followup': (.075, .810, .130, .085),
        'pregnancy_loss': (.230, .810, .130, .085),
        'children_without_pft_records': (.145, .610, .265, .080),
        'pft_children_without_descriptive_measurements': (.145, .435, .265, .110),
        'no_tb_before_five': (.270, .220, .30, .085),
        'tb_before_five': (.675, .220, .30, .085),
        'no_recorded_tb': (.120, .065, .20, .100),
        'age_five_or_later': (.360, .065, .20, .100),
        'infancy': (.610, .065, .20, .100),
        'age_one_to_under_five': (.850, .065, .20, .100),
        'spirometry': (.835, .850, .275, .155),
        'mechanics': (.835, .660, .275, .120),
        'within_breath': (.835, .500, .275, .120),
        'x6': (.835, .365, .275, .075),
    }
    nodes = {n['id']: n for n in panel['nodes']}
    if set(nodes) != set(positions):
        raise ValueError('The selected flow node inventory differs from the declared layout.')
    ax = canvas.axes(left, bottom, width, height)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for parent, child in panel['edges']:
        x1, y1, w1, h1 = positions[parent]
        x2, y2, w2, h2 = positions[child]
        role = nodes[child]['role']
        if role == 'outcome':
            # Shared dashed trunk denotes overlapping analytical samples.
            start, end = (x1 + w1 / 2, y1), (x2 - w2 / 2, y2)
            mid = .67
            ax.plot([start[0], mid, mid, end[0]], [start[1], start[1], end[1], end[1]],
                    color='#A6B1BA', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
            continue
        if role == 'exclusion' and x2 < x1:
            start, end = (x1 - w1 / 2, y1), (x2 + w2 / 2, y2)
            mid = (start[0] + end[0]) / 2
            vertices = [start, (mid, start[1]), (mid, end[1]), end]
        else:
            start, end = (x1, y1 - h1 / 2), (x2, y2 + h2 / 2)
            mid = (start[1] + end[1]) / 2
            vertices = [start, (start[0], mid), (end[0], mid), end]
        path = DrawPath(vertices, [DrawPath.MOVETO] + [DrawPath.LINETO] * (len(vertices) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=10,
                     linewidth=.85, color='#8494A1', zorder=0))
    canvas.label(left + width * .835, bottom + height * .970, 'Outcome-specific samples (overlap)',
                 size=10, bold=True, ha='center', width=width * .285)
    for ident, node in nodes.items():
        x, y, w, h = positions[ident]
        unavailable = node['status'] == 'unavailable'
        role = node['role']
        fill = '#F3F5F6' if unavailable or role.startswith('exclusion') else '#EAF2F7'
        if ident == 'combined_eligible_lung_function': fill = '#D8E8F2'
        if ident == 'tb_before_five': fill = '#C3DFEE'
        border = '#A7B2BA' if unavailable else '#5D849C'
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
            boxstyle='round,pad=0.005,rounding_size=0.009', facecolor=fill, edgecolor=border,
            linewidth=.9, linestyle='--' if unavailable else '-', zorder=2))
        size = 8.5 if role == 'exclusion_detail' else 9.2 if role == 'outcome' else 9.5
        wrapped = canvas.wrap(node['text'], width * w - .14, size, role in ('main', 'stratum'))
        lines = len(wrapped.splitlines())
        if lines * size * 1.22 / 72 > height * h - .06:
            raise ValueError('Flow box is too small for its supplied text: ' + ident)
        canvas.label(left + width * x, bottom + height * y, wrapped, size=size,
                     bold=role in ('main', 'stratum'), ha='center', va='center')
    canvas.panel_metadata.append({'id': panel['id'], 'kind': 'flow', 'nodes': len(nodes),
        'edges': len(panel['edges']), 'status_counts': dict(Counter(n['status'] for n in nodes.values())),
        'unavailable_nodes': [n['id'] for n in nodes.values() if n['status'] == 'unavailable'],
        'counts_recalculated_by_renderer': False, 'outcome_samples_overlap': panel['outcome_samples_overlap']})


def _schedule(canvas, panel, box):
    from matplotlib.patches import FancyBboxPatch
    left, bottom, width, height = box
    rows, columns = panel['row_order'], panel['column_order']
    cells = {(c['row'], c['column']): c for c in panel['cells']}
    if set(cells) != {(r, c) for r in rows for c in columns}:
        raise ValueError('The schedule requires every declared row-column cell, including empty cells.')
    label_width = width * .265
    gx, gw = left + label_width, width - label_width
    dx = gw / len(columns)
    top = bottom + height - .85
    row_height = (height - 1.7) / len(rows)
    ax = canvas.axes(left, bottom, width, height)
    ax.axis('off'); ax.set_xlim(0, width); ax.set_ylim(0, height)
    colors = ['#2A788E', '#246B96', '#7D8994', '#C8792C']
    for group in panel['column_groups']:
        indices = [columns.index(c) for c in group['columns']]
        start = label_width + min(indices) * dx + .02
        end = label_width + (max(indices) + 1) * dx - .02
        ax.add_patch(FancyBboxPatch((start, height - .55), end - start, .40,
            boxstyle='round,pad=0.015,rounding_size=.06', facecolor='#EEF2F5', edgecolor='none'))
        canvas.label(left + (start + end) / 2, bottom + height - .35, group['label'],
                     size=10, ha='center', va='center')
    for i, row in enumerate(rows):
        y = top - (i + .5) * row_height
        canvas.label(left, y, panel['row_labels'][row], size=10.5, bold=True,
                     width=label_width - .2, va='center')
        for j, column in enumerate(columns):
            cell = cells[row, column]
            represented = cell['represented']
            if type(represented) is not bool:
                raise ValueError('Schedule presence must be an explicit boolean.')
            x = label_width + j * dx + dx * .13
            yy = y - bottom - row_height * .30
            color = colors[i % len(colors)] if represented else '#F6F8F9'
            ax.add_patch(FancyBboxPatch((x, yy), dx * .74, row_height * .60,
                boxstyle='round,pad=0.005,rounding_size=.055', facecolor=color,
                edgecolor='#DAE1E6', linewidth=.65))
            if represented:
                ax.plot(x + dx * .37, yy + row_height * .30, marker='o', markersize=3.4,
                        color='white', linestyle='none')
    for j, column in enumerate(columns):
        canvas.label(gx + (j + .5) * dx, bottom + .72, panel['column_labels'][column],
                     size=9, ha='center', va='top', width=dx - .05)
    canvas.label(gx + gw / 2, bottom + .20, 'Nominal study visit', size=10, ha='center', va='bottom')
    canvas.panel_metadata.append({'id': panel['id'], 'kind': 'schedule', 'rows': len(rows),
        'columns': len(columns), 'cells': len(cells), 'represented_cells': sum(c['represented'] for c in cells.values()),
        'display': 'presence only', 'status_counts': dict(Counter(c['status'] for c in cells.values())),
        'row_order': rows, 'column_order': columns})


def _build_figure(spec, fonts):
    # Page size follows the amount of content. Captions stay on the same page.
    ident, panels = spec['id'], spec['panels']
    width = 12 if spec['kind'] in ('flow', 'schedule') else 11 if ident == 'main_figure_3' else 10.5
    content_height = {
        'main_figure_1': 7.4, 'main_figure_2': 8.8, 'main_figure_3': 4.1,
        'appendix_figure_s1': 8.8, 'appendix_figure_s2': 4.2,
        'appendix_figure_s3': 8.1, 'appendix_figure_s4': 4.2,
        'appendix_figure_s5': 3.7, 'appendix_figure_s6': 3.4,
    }[ident]
    # Measure text on the actual Arial canvas before committing the page height.
    canvas = _Canvas(width, 12, fonts)
    title = canvas.wrap(spec['title'], width - 1, 13, True)
    caption = canvas.wrap(spec['caption']['text'], width - 1, 9.2)
    title_height = len(title.splitlines()) * .235 + .35
    caption_height = len(caption.splitlines()) * .163 + .22
    legend_height = 0
    if spec['kind'] == 'forest':
        labels = {p['series_styles'][s]['label'] for p in panels for s in p['series_order']}
        if len(labels) > 1:
            legend_height = .65 if any(len(s) > 48 for s in labels) or sum(map(len, labels)) > 100 else .4
    height = .5 + title_height + legend_height + content_height + .35 + caption_height + .35
    canvas.fig.set_size_inches(width, height)
    canvas.height = height
    top = height - .4
    canvas.label(.5, top, title, size=13, bold=True)
    content_top = top - title_height - legend_height
    if legend_height:
        _legend(canvas, panels, top - title_height + .06, width=width - 1)
    content_bottom = content_top - content_height
    if spec['kind'] == 'trajectory':
        gap_x, gap_y = .55, .30
        cell_width = (width - 1 - gap_x) / 2
        cell_height = (content_height - gap_y) / 2
        for i, panel in enumerate(panels):
            _trajectory(canvas, panel, (.5 + (i % 2) * (cell_width + gap_x),
                content_top - (i // 2 + 1) * cell_height - (i // 2) * gap_y, cell_width, cell_height))
    elif spec['kind'] == 'flow':
        _flow(canvas, panels[0], (.4, content_bottom, width - .8, content_height))
    elif spec['kind'] == 'schedule':
        _schedule(canvas, panels[0], (.5, content_bottom, width - 1, content_height))
    elif ident in ('main_figure_3', 'appendix_figure_s4'):
        gap = .55
        cell_width = (width - 1 - gap) / 2
        for i, panel in enumerate(panels):
            box = (.5 + i * (cell_width + gap), content_bottom, cell_width, content_height)
            if panel['orientation'] == 'vertical':
                _forest_vertical(canvas, panel, box)
            else:
                _forest_horizontal(canvas, panel, box, compact=True)
    else:
        gap = .2
        panel_height = (content_height - gap * (len(panels) - 1)) / len(panels)
        for i, panel in enumerate(panels):
            box = (.5, content_top - (i + 1) * panel_height - i * gap, width - 1, panel_height)
            _forest_horizontal(canvas, panel, box)
            if i:
                canvas.line(.5, box[1] + box[3] + .1, width - .5, box[1] + box[3] + .1)
    caption_top = content_bottom - .23
    canvas.line(.5, caption_top + .10, width - .5, caption_top + .10)
    canvas.label(.5, caption_top, caption, size=9.2)
    return canvas


def render_figures(data, destination, fonts, *, dpi=600):
    """Write nine vector PDFs, nine 600-dpi PNGs and one nine-page PDF.

    ``data`` is the validated output of figure_data.build_data. ``fonts`` names
    the actual Arial regular/bold font files. Returned metadata is separate from
    the nineteen image files and does not contain its own hash.
    """
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import font_manager, ft2font
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import rc_context
    from .render_document import _font
    if type(dpi) is not int or dpi != 600:
        raise ValueError('Publication PNGs require exactly 600 dpi.')
    if data.get('schema_version') != 1 or [f['id'] for f in data['figures']] != list(FIGURE_IDS):
        raise ValueError('The nine selected figures are required in their publication order.')
    before = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    filenames = [ident + suffix for ident in FIGURE_IDS for suffix in ('.pdf', '.png')] + ['publication_figures.pdf']
    if any((destination / name).exists() for name in filenames):
        raise ValueError('Figure outputs already exist; an existing publication is never overwritten.')
    checked_fonts = {}
    for key, bold in (('arial', False), ('arial_bold', True)):
        path = Path(fonts[key])
        _font(path.read_bytes(), bold)
        face = ft2font.FT2Font(str(path))
        if face.family_name != 'Arial' or ('bold' in face.style_name.lower()) != bold:
            raise ValueError('Actual Arial regular and Arial Bold font files are required.')
        font_manager.fontManager.addfont(str(path))
        checked_fonts[key] = {'sha256': _hash(path), 'family': face.family_name, 'style': face.style_name}
    rc = {'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'font.size': 10,
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'axes.unicode_minus': False, 'text.usetex': False,
          'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial',
          'mathtext.bf': 'Arial:bold', 'mathtext.bfit': 'Arial:bold',
          'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'mathtext.cal': 'Arial',
          'mathtext.fallback': None, 'mathtext.default': 'regular',
          'savefig.facecolor': 'white', 'savefig.edgecolor': 'white'}
    pdfmeta = {'Title': 'DCHS lung function publication figures', 'Author': '',
               'Subject': 'Figures from verified aggregate results', 'Keywords': '',
               'Creator': 'DCHS lung function reporting', 'Producer': 'Matplotlib',
               'CreationDate': None, 'ModDate': None}
    result = {'schema_version': 1, 'renderer': {'library': 'Matplotlib', 'version': matplotlib.__version__},
              'dpi': dpi, 'fonts': checked_fonts, 'figures': [], 'combined_pdf': 'publication_figures.pdf',
              'scientific_values_modified': False, 'files': {}}
    with rc_context(rc), PdfPages(destination / 'publication_figures.pdf', metadata=pdfmeta) as combined:
        for spec in data['figures']:
            canvas = _build_figure(spec, fonts)
            audit = canvas.audit()
            ident = spec['id']
            metadata = {**pdfmeta, 'Title': spec['title']}
            canvas.fig.savefig(destination / (ident + '.pdf'), format='pdf', metadata=metadata)
            combined.savefig(canvas.fig)
            canvas.fig.savefig(destination / (ident + '.png'), format='png', dpi=dpi,
                               metadata={'Software': 'DCHS lung function reporting'})
            # Report the rasterizer's actual dimensions; floating-point inch
            # arithmetic can differ from its pixel rounding by one pixel.
            with (destination / (ident + '.png')).open('rb') as image:
                header = image.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
                raise ValueError('Rendered PNG is missing its dimension header.')
            pixels = [int.from_bytes(header[16:20], 'big'), int.from_bytes(header[20:24], 'big')]
            result['figures'].append({'id': ident, 'pdf': ident + '.pdf', 'png': ident + '.png',
                'width_inches': canvas.width, 'height_inches': canvas.height,
                'pixel_dimensions': pixels,
                'panels': canvas.panel_metadata, 'caption_text': spec['caption']['text'],
                'layout_checks': audit})
            canvas.fig.clear()
    if json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) != before:
        raise ValueError('Figure rendering changed the supplied scientific data.')
    result['files'] = {name: _hash(destination / name) for name in filenames}
    return result
