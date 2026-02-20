"""
Функции визуализации из Streamlit app.py
Точные копии без изменений
"""
import io
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch, Pitch, PyPizza
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from contextlib import contextmanager
import matplotlib.patheffects as path_effects

# Цвета из app.py
PitchColor = "#f5f6fc"
BackgroundColor = "#381d54"
PitchLineColor = "Black"
TextColor = "White"

@contextmanager
def safe_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)

def normalize_position(pos: str) -> str:
    """Normalize position strings into standard buckets."""
    if pos is None:
        return pos
    pos = str(pos)
    if pos in ['RWB', 'LWB']:
        return pos
    if 'CM(2)' in pos:
        return 'CM(2)'
    elif 'CM(3)' in pos:
        return 'CM(3)'
    elif 'DM(2)' in pos or 'DM(3)' in pos:
        return 'DM(23)'
    elif 'CB(2)' in pos:
        return 'CB(2)'
    elif 'CB(3)' in pos:
        return 'CB(3)'
    elif 'AM(2)' in pos:
        return 'AM(2)'
    elif 'CF(2)' in pos:
        return 'CF(2)'
    elif 'LMW' in pos or 'LW' in pos or 'LM' in pos:
        return 'LW'
    elif 'RMW' in pos or 'RW' in pos or 'RM' in pos:
        return 'RW'
    elif pos in ['DM', 'AM', 'CF', 'LB', 'RB', 'GK']:
        return pos
    else:
        return pos


def create_shot_map(matchdata, playername, team_choice, position):
    """
    ТОЧНАЯ копия функции карты ударов из app.py
    Возвращает BytesIO с PNG изображением
    """
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Debug logging
    logger.info(f"create_shot_map called: player={playername}, team={team_choice}, position={position}")
    logger.info(f"Matchdata shape: {matchdata.shape}")
    logger.info(f"Unique positions in matchdata: {matchdata['playing_position'].unique() if 'playing_position' in matchdata.columns else 'N/A'}")
    
    # Фильтруем удары игрока
    playershots = matchdata.loc[
        (matchdata['playerName'] == playername) &
        (matchdata['team_name'] == team_choice) &
        (matchdata['playing_position'] == position)
    ].copy()
    
    logger.info(f"Filtered playershots shape: {playershots.shape}")
    
    playershots = playershots.loc[playershots['shotType'].notna()]
    
    logger.info(f"After shotType filter: {playershots.shape}")
    
    if playershots.empty:
        logger.warning(f"No shot data found for {playername} at {position}")
        return None
    
    shotmaptar2 = playershots.loc[playershots['typeId']=='Attempt Saved']
    shotmaptar2 = shotmaptar2.loc[shotmaptar2['expectedGoalsOnTarget']>0]
    
    shotmapbk2 = playershots.loc[playershots['typeId']=='Attempt Saved']
    shotmapbk2 = shotmapbk2.loc[shotmapbk2['expectedGoalsOnTarget']== 0]
    
    shotmapoff2 = playershots.loc[playershots['typeId'] == 'Miss']
    
    goalmap3 = playershots.loc[
        (playershots['typeId'] == 'Goal') &
        (playershots['typeId'] != 'Own Goal')
    ]
    
    num_goals = len(goalmap3)
    num_shots = len(shotmaptar2) + len(shotmapbk2) + len(shotmapoff2) + len(goalmap3)
    shots_on_target = len(shotmaptar2) + len(goalmap3)
    shot_conversion_rate = round((shots_on_target / num_shots) * 100, 1) if num_shots > 0 else 0
    goal_conversion_rate = round((num_goals / num_shots) * 100, 1) if num_shots > 0 else 0
    xg_sum = round(playershots['expectedGoals'].sum(), 2)
    xgot_sum = round(playershots['expectedGoalsOnTarget'].sum(), 2)
    
    # Создаем визуализацию
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.set_facecolor("#FAF9F6")
    
    pitch_left = VerticalPitch(
        pitch_type='opta',
        half=True,
        pitch_color="#FFFFFF",
        line_color="#000000",
        linewidth=2.5
    )
    pitch_left.draw(ax=ax)
    
    def get_marker_size(expectedGoals, scale_factor=1200):
        return expectedGoals * scale_factor
    
    # Goals - зеленый заполненный
    pitch_left.scatter(
        goalmap3.x, goalmap3.y,
        s=get_marker_size(goalmap3.expectedGoals),
        ax=ax, edgecolor='#2ECC71', facecolor='#2ECC71', marker='o',
        linewidths=2, alpha=0.8, zorder=4,
        label='Goal'
    )
    
    # Shot on Target - синий заполненный
    pitch_left.scatter(
        shotmaptar2.x, shotmaptar2.y,
        s=get_marker_size(shotmaptar2.expectedGoals),
        ax=ax, edgecolor='#0066CC', facecolor='#0066CC', marker='o',
        linewidths=2, alpha=0.7, zorder=3,
        label='Shot on Target'
    )
    
    # Shot Blocked - оранжевая окружность (пустая)
    pitch_left.scatter(
        shotmapbk2.x, shotmapbk2.y,
        s=get_marker_size(shotmapbk2.expectedGoals),
        ax=ax, edgecolor='#FF6B35', facecolor='none', marker='o',
        linewidths=2, alpha=0.8, zorder=2,
        label='Shot Blocked'
    )
    
    # Shot off Target - красная окружность (пустая)
    pitch_left.scatter(
        shotmapoff2.x, shotmapoff2.y,
        s=get_marker_size(shotmapoff2.expectedGoals),
        ax=ax, edgecolor='#FF0000', facecolor='none', marker='o',
        linewidths=2, alpha=0.8, zorder=1,
        label='Shot off Target'
    )
    
    ax.set_title(
        f"{playername} - xG Shot Map as {position}",
        fontsize=16, color="#000000", weight='bold', pad=15
    )
    
    handles, labels = ax.get_legend_handles_labels()
    legend_markers = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#2ECC71',
                   markeredgecolor='#2ECC71', markersize=12, markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#0066CC',
                   markeredgecolor='#0066CC', markersize=12, markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='#FF6B35', markersize=12, markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='#FF0000', markersize=12, markeredgewidth=2)
    ]
    
    ax.legend(
        legend_markers, labels,
        facecolor='white', framealpha=0.9, handlelength=2, edgecolor='#000000',
        bbox_to_anchor=(0.034, 0.96), fontsize=9, loc='upper left'
    )
    
    # Stats box
    stats_x, stats_y = 99, 73
    line_height = 1.8
    ax.text(stats_x, stats_y, f'Goals: {num_goals}', ha='left', fontsize=10, color='#000000', weight='bold')
    ax.text(stats_x, stats_y - line_height, f'xG: {xg_sum}', ha='left', fontsize=10, color='#000000')
    ax.text(stats_x, stats_y - line_height*2, f'xGOT: {xgot_sum}', ha='left', fontsize=10, color='#000000')
    ax.text(stats_x, stats_y - line_height*3, f'Shots: {num_shots}', ha='left', fontsize=10, color='#000000')
    ax.text(stats_x, stats_y - line_height*4, f'On Target: {shots_on_target}', ha='left', fontsize=10, color='#000000')
    ax.text(stats_x, stats_y - line_height*5, f'Accuracy: {shot_conversion_rate}%', ha='left', fontsize=10, color='#000000', weight='bold')
    ax.text(stats_x, stats_y - line_height*6, f'Conversion: {goal_conversion_rate}%', ha='left', fontsize=10, color='#000000', weight='bold')
    
    # Сохраняем в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#FAF9F6')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def filter_player_data(matchdata, playername, team_choice, position):
    """Helper function to filter matchdata by player, team, and position."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"filter_player_data: player={playername}, team={team_choice}, position={position}")
    logger.info(f"Matchdata shape before filter: {matchdata.shape}")
    
    if position == "All Positions":
        result = matchdata.loc[
            (matchdata['playerName'] == playername) &
            (matchdata['team_name'] == team_choice)
        ].copy()
    else:
        result = matchdata.loc[
            (matchdata['playerName'] == playername) &
            (matchdata['playing_position'] == position) &
            (matchdata['team_name'] == team_choice)
        ].copy()
    
    logger.info(f"Filtered data shape: {result.shape}")
    if result.empty:
        logger.warning(f"No data found! Available positions for {playername}: {matchdata[matchdata['playerName']==playername]['playing_position'].unique() if 'playing_position' in matchdata.columns else 'N/A'}")
    
    return result

def format_position_display(position):
    """Helper function to format position for display in titles."""
    if position == "All Positions":
        return "All Positions"
    else:
        return f"as {position}"

def create_pass_and_carry_sonar(matchdata, playername, team_choice, position):
    """
    ТОЧНАЯ копия функции sonar из app.py
    Возвращает BytesIO с PNG изображением
    """
    from matplotlib.patches import Wedge
    
    # Фильтруем данные
    base = filter_player_data(matchdata, playername, team_choice, position)
    
    # Дополнительные фильтры
    base = base.loc[
        (base['throwin'] != 1) &
        (base['corner'] != 1) &
        (base['freekick'] != 1) &
        (base['goalkick'] != 1)
    ].copy()

    # Убираем kick-off
    mask1 = ~((base['timeMin'] == 0) & (base['timeSec'] == 0))
    mask2 = ~((base['timeMin'] == 45) & (base['timeSec'] == 0))
    base = base[mask1 & mask2]

    passingdata = base.loc[base['typeId'] == 'Pass'].copy()
    carryingdata = base.loc[base['typeId'] == 'Carry'].copy()

    if passingdata.empty and carryingdata.empty:
        return None

    # Настройки
    x_bins = np.linspace(0, 100, 6)
    y_bins = np.linspace(0, 100, 6)

    def plot_sonar(ax, cx, cy, angles, bins=12, max_radius=8, color='#000000'):
        edges = np.linspace(0, 360, bins + 1)
        counts, _ = np.histogram(angles, bins=edges)
        radii = (counts / counts.max() * max_radius) if counts.max() else np.zeros_like(counts)

        for i in range(bins):
            if radii[i] > 0:
                wedge = Wedge(
                    center=(cy, cx),
                    r=radii[i],
                    theta1=edges[i],
                    theta2=edges[i + 1],
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0
                )
                ax.add_patch(wedge)

    def add_grid(ax):
        for xb in x_bins:
            ax.add_line(Line2D([0, 100], [xb, xb], color='#000000', ls='--', lw=.7, alpha=.25))
        for yb in y_bins:
            ax.add_line(Line2D([yb, yb], [0, 100], color='#000000', ls='--', lw=.7, alpha=.25))

    def sonar(ax, data, title, sonar_color='#000000'):
        if data.empty:
            ax.set_title(f"{title} (No data)", color="#000000")
            return

        df = data.copy()
        df['hx'] = df['end_y'] - df['y']
        df['vy'] = df['end_x'] - df['x']
        df['angle'] = (np.degrees(np.arctan2(df['vy'], df['hx'])) + 360) % 360

        df['row'] = pd.cut(df['x'], bins=x_bins, labels=False)
        df['col'] = pd.cut(df['y'], bins=y_bins, labels=False)

        add_grid(ax)

        for r in range(5):
            for c in range(5):
                cell = df[(df['row'] == r) & (df['col'] == c)]
                if not len(cell):
                    continue

                cx = (x_bins[r] + x_bins[r+1]) / 2
                cy = (y_bins[c] + y_bins[c+1]) / 2

                plot_sonar(ax, cx, cy, cell['angle'], max_radius=7, color=sonar_color)

        ax.set_title(title, color="#000000")

    # Создаем фигуру
    pitch = VerticalPitch(
        pitch_type='opta',
        goal_type='box',
        pitch_color="#FFFFFF",
        line_color="#000000",
        linewidth=2.5
    )

    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(12, 9))
    fig.set_facecolor("#FAF9F6")

    position_display = format_position_display(position)
    sonar(axes[0], passingdata, f"{playername} - Passing Sonars {position_display}", sonar_color='#FF6B35')
    sonar(axes[1], carryingdata, f"{playername} - Carrying Sonars {position_display}", sonar_color='#004E89')
    
    # Add PASSING/CARRYING labels
    fig.text(0.25, 0.95, 'PASSING', ha='center', fontsize=14, color='#FF6B35', weight='bold')
    fig.text(0.75, 0.95, 'CARRYING', ha='center', fontsize=14, color='#004E89', weight='bold')
    
    fig.text(0.5, 0.02, 'Created: Novvi | Data: WTanalysis', 
             ha='center', fontsize=8, color='#000000', alpha=0.7)
    
    # Сохраняем
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#FAF9F6')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_player_actions(matchdata, playername, team_choice, position):
    """
    ТОЧНАЯ копия функции Player Actions из app.py
    Возвращает BytesIO с PNG изображением
    """
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import ConvexHull
    from scipy.stats import gaussian_kde
    
    def _can_hull(points):
        """Check if we can safely compute a 2-D convex hull."""
        if len(points) < 3:
            return False
        uniq = np.unique(points, axis=0)
        if len(uniq) < 3:
            return False
        p1, p2, p3 = uniq[:3]
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return area > 1e-6
    
    # Фильтруем события игрока
    playerevents = filter_player_data(matchdata, playername, team_choice, position)
    
    # Получаем пасы
    playerrecpass = matchdata.loc[
        (matchdata["playerName"] == playername) &
        (matchdata["team_name"] == team_choice)
    ].copy()
    if position != "All Positions":
        playerrecpass = playerrecpass.loc[playerrecpass['pass_recipient_position'] == position]
    playerrecpass = playerrecpass.loc[playerrecpass['outcome'] == "Successful"]
    playerrecpass = playerrecpass[
        (playerrecpass['x'].between(1, 99)) &
        (playerrecpass['y'].between(1, 99)) &
        (playerrecpass['end_x'].between(1, 99)) &
        (playerrecpass['end_y'].between(1, 99))
    ]
    
    # Defensive events
    defeven = ['Tackle', 'Aerial', 'Challenge', 'Interception', 'Save', 'Clearance', 'Ball recovery']
    defensiveevents = playerevents[playerevents['typeId'].isin(defeven)].copy()
    defensiveevents = defensiveevents[~((defensiveevents['typeId'] == 'Aerial') & (defensiveevents['x'] >= 50))]
    
    # Attacking events
    atteven = ['Take on', 'Miss', 'Attempt Saved', 'Goal', 'Aerial', 'Post']
    attackingevents = playerevents[
        (playerevents['typeId'].isin(atteven)) |
        (playerevents.get('keyPass', 0) == 1) |
        (playerevents.get('assist', 0) == 1) |
        (playerevents.get('progressive_carry', 'No') == "Yes")
    ].copy()
    attackingevents = attackingevents[~((attackingevents['typeId'] == 'Aerial') & (attackingevents['x'] < 50))]
    attackingevents = attackingevents[~(attackingevents.get('corner', 0) == 1)]
    
    # Создаем фигуру
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.25), facecolor="#FAF9F6")
    plt.subplots_adjust(wspace=.05)
    
    pitch = VerticalPitch(pitch_type='opta', pitch_color="#FFFFFF", line_color="#000000", linewidth=2.5)
    
    pitch.draw(ax=axes[0])
    axes[0].set_title(f'{playername} - Attacking Events as {position}', fontsize=10, color="#000000", weight='bold', pad=8)
    
    pitch.draw(ax=axes[1])
    axes[1].set_title(f'{playername} - Defensive Events as {position}', fontsize=10, color="#000000", weight='bold', pad=8)
    
    pitch.draw(ax=axes[2])
    axes[2].set_title(f'{playername} - Pass Receptions as {position}', fontsize=10, color="#000000", weight='bold', pad=8)
    
    # KDE grid
    x_grid, y_grid = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
    
    # Attacking events
    points_att = np.array([(row['y'], row['x']) for _, row in attackingevents.iterrows()])
    if len(points_att) > 3:
        kde_att = gaussian_kde(points_att.T)
        density_att = kde_att(np.vstack([x_grid.ravel(), y_grid.ravel()]))
        max_idx = np.argmax(density_att)
        max_x = x_grid.ravel()[max_idx]
        max_y = y_grid.ravel()[max_idx]
        radius = 15
        points_within = points_att[((points_att[:, 0] - max_x)**2 + (points_att[:, 1] - max_y)**2) < radius**2]
        if _can_hull(points_within):
            hull = ConvexHull(points_within)
            hull_patch = MplPolygon(
                np.column_stack([points_within[hull.vertices, 0], points_within[hull.vertices, 1]]),
                closed=True, edgecolor="#8B00FF", facecolor="#8B00FF", linewidth=2, alpha=0.2
            )
            axes[0].add_patch(hull_patch)
    
    for _, row in attackingevents.iterrows():
        axes[0].plot(row['y'], row['x'], marker='o', markerfacecolor='#8B00FF', markeredgecolor='#8B00FF', markersize=3)
    
    # Defensive events
    points_def = np.array([(row['y'], row['x']) for _, row in defensiveevents.iterrows()])
    if len(points_def) > 3:
        kde_def = gaussian_kde(points_def.T)
        density_def = kde_def(np.vstack([x_grid.ravel(), y_grid.ravel()]))
        max_idx = np.argmax(density_def)
        max_x = x_grid.ravel()[max_idx]
        max_y = y_grid.ravel()[max_idx]
        radius = 15
        points_within = points_def[((points_def[:, 0] - max_x)**2 + (points_def[:, 1] - max_y)**2) < radius**2]
        if _can_hull(points_within):
            hull = ConvexHull(points_within)
            hull_patch = MplPolygon(
                np.column_stack([points_within[hull.vertices, 0], points_within[hull.vertices, 1]]),
                closed=True, edgecolor="#FF0000", facecolor="#FF0000", linewidth=2, alpha=0.2
            )
            axes[1].add_patch(hull_patch)
    
    for _, row in defensiveevents.iterrows():
        axes[1].plot(row['y'], row['x'], marker='o', markerfacecolor='#FF0000', markeredgecolor='#FF0000', markersize=3)
    
    # Pass receptions
    points_rec = np.array([(row['end_y'], row['end_x']) for _, row in playerrecpass.iterrows()])
    if len(points_rec) > 3:
        kde_rec = gaussian_kde(points_rec.T)
        density_rec = kde_rec(np.vstack([x_grid.ravel(), y_grid.ravel()]))
        max_idx = np.argmax(density_rec)
        max_x = x_grid.ravel()[max_idx]
        max_y = y_grid.ravel()[max_idx]
        radius = 15
        points_within = points_rec[((points_rec[:, 0] - max_x)**2 + (points_rec[:, 1] - max_y)**2) < radius**2]
        if _can_hull(points_within):
            hull = ConvexHull(points_within)
            hull_patch = MplPolygon(
                np.column_stack([points_within[hull.vertices, 0], points_within[hull.vertices, 1]]),
                closed=True, edgecolor="#0000FF", facecolor="#0000FF", linewidth=2, alpha=0.2
            )
            axes[2].add_patch(hull_patch)
    
    for _, row in playerrecpass.iterrows():
        axes[2].plot(row['end_y'], row['end_x'], marker='o', markerfacecolor='#0000FF', markeredgecolor='#0000FF', markersize=3)
    
    # Текст
    axes[0].text(60, -10, f'{playername} - {team_choice}', ha='left', fontsize=16, color="#000000", fontweight='bold')
    axes[2].text(1, -22, 'Data: WTanalysis | Created: Novvi', ha='right', fontsize=10, color="#000000")
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#FAF9F6')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_zone_dominance(matchdata, playername, team_choice, position):
    """
    ТОЧНАЯ копия функции Zone Dominance из app.py
    Возвращает BytesIO с PNG изображением
    """
    from mplsoccer import Pitch
    import matplotlib.patheffects as path_effects
    
    touch_events = ['Pass', 'Carry', 'Take On', 'Miss', 'Attempt Saved', 'Goal', 
                    'Post', 'Tackle', 'Interception', 'Clearance', 'Ball recovery', 'Aerial']
    
    player_touches = filter_player_data(matchdata, playername, team_choice, position)
    player_touches = player_touches.loc[player_touches['typeId'].isin(touch_events)].copy()
    
    if player_touches.empty:
        return None
    
    pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black', linewidth=2.5, line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('white')
    
    bin_statistic = pitch.bin_statistic(
        player_touches['x'], player_touches['y'], 
        statistic='count', bins=(6, 4), normalize=True
    )
    
    pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='black')
    
    path_eff = [path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()]
    pitch.label_heatmap(
        bin_statistic, color='black', fontsize=16, ax=ax, 
        ha='center', va='center', str_format='{:.1%}', path_effects=path_eff
    )
    
    ax.set_title(f'{playername} - Zone Dominance Map ({format_position_display(position)})', 
                 fontsize=18, weight='bold', pad=20, color='black')
    
    total_touches = len(player_touches)
    ax.text(50, -5, f'Total Touches: {total_touches} | Created: Novvi | Data: WTanalysis',
            ha='center', fontsize=11, color='black', transform=ax.transData)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_heatmap(matchdata, playername, team_choice, position):
    """
    ТОЧНАЯ копия функции Touch Heatmap из app.py
    Возвращает BytesIO с PNG изображением
    """
    from mplsoccer import Pitch
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    
    touch_events = ['Pass', 'Carry', 'Take On', 'Miss', 'Attempt Saved', 'Goal', 
                    'Post', 'Tackle', 'Interception', 'Clearance', 'Ball recovery', 'Aerial']
    
    player_touches = filter_player_data(matchdata, playername, team_choice, position)
    player_touches = player_touches.loc[player_touches['typeId'].isin(touch_events)].copy()
    
    if player_touches.empty or len(player_touches) < 5:
        return None
    
    # Convert to numeric
    player_touches['x'] = pd.to_numeric(player_touches['x'], errors='coerce')
    player_touches['y'] = pd.to_numeric(player_touches['y'], errors='coerce')
    player_touches = player_touches.dropna(subset=['x', 'y'])
    
    if len(player_touches) < 5:
        return None
    
    x = player_touches['x'].values
    y = player_touches['y'].values
    
    # Use horizontal Pitch (not VerticalPitch!)
    pitch = Pitch(
        pitch_type='opta',
        line_zorder=2,
        pitch_color='#22312b',
        line_color='#efefef'
    )
    
    fig, ax = pitch.draw(figsize=(6.6, 4.125))
    fig.set_facecolor('#22312b')
    
    # Создаем кастомную цветовую карту
    colors = ['#15242e', '#4393c4', '#e8f4f8']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=10)
    
    # Bin statistic with Gaussian smoothing
    bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    
    # Draw heatmap
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, edgecolors='#22312b')
    
    # Add colorbar
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    import matplotlib.pyplot as plt
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
    
    ax.set_title(f'{playername} - Touch Heatmap ({format_position_display(position)})', 
                 fontsize=12, pad=10, color='#efefef')
    
    total_touches = len(player_touches)
    ax.text(50, -5, f'Total Touches: {total_touches} | Created: Novvi | Data: WTanalysis',
            ha='center', fontsize=8, color='#efefef', alpha=0.7)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#22312b')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_pizza_chart(player_stats, player_name, team_name, position, color1="#1E88E5", color2="#FFC107"):
    """
    Создать Pizza Chart для игрока
    Возвращает BytesIO с PNG изображением
    """
    import logging
    logger = logging.getLogger(__name__)
    
    from mplsoccer import PyPizza
    
    logger.info(f"create_pizza_chart called: player={player_name}, team={team_name}, position={position}")
    
    # Сначала пробуем найти данные с нормализованной позицией
    player_data = player_stats[
        (player_stats['player_name'] == player_name) &
        (player_stats['team_name'] == team_name) &
        (player_stats['position_group'] == position)
    ]
    
    # Если не нашли, пробуем найти с любой позицией и взять ту, где больше всего минут
    if player_data.empty:
        logger.info(f"No data with exact position {position}, trying to find any position")
        player_data_all = player_stats[
            (player_stats['player_name'] == player_name) &
            (player_stats['team_name'] == team_name)
        ]
        
        if player_data_all.empty:
            logger.warning(f"No data found for {player_name} at {team_name}")
            return None
        
        # Берем позицию с наибольшим количеством минут
        player_data = player_data_all.sort_values('minutes_played', ascending=False).head(1)
        actual_position = player_data.iloc[0]['position_group']
        logger.info(f"Using position {actual_position} instead of {position}")
        position = actual_position
    
    logger.info(f"Player data shape: {player_data.shape}")
    logger.info(f"Using position: {position}")
    
    # Метрики для разных позиций
    if position in ['CF', 'LW', 'RW', 'AM', 'CF(2)', 'AM(2)', 'RMW', 'LMW']:
        metrics = [
            'shots_per_90', 'shot_accuracy', 'xG_per_90', 'goals_per_90',
            'shot_quality', 'prog_carries_per_90', 'carrying_yards_per_90',
            'dribbles_per_90', 'successful_attacking_actions_per_90',
            'pass_completion_final_third', 'keyPasses_per_90',
            'xA_per_90', 'assists_per_90', 'threat_value_per_90'
        ]
        labels = [
            'Shots', 'Shot Acc%', 'xG', 'Goals', 'Shot Quality',
            'Prog Carries', 'Carry Yards', 'Dribbles', 'Att Actions',
            'Pass Acc F3rd%', 'Key Passes', 'xA', 'Assists', 'Impact'
        ]
    elif position in ['CM(2)', 'CM(3)', 'DM(23)', 'DM', 'AM']:
        metrics = [
            'goals_per_90', 'xG_per_90', 'assists_per_90', 'keyPasses_per_90',
            'pass_completion', 'pass_completion_final_third', '%_passes_are_progressive',
            'prog_carries_per_90', 'total_threat_created_per_90',
            'defensive_actions_per_90', 'successful_defensive_actions_per_90',
            'interceptions_per_90', 'total_threat_prevented_per_90', 'threat_value_per_90'
        ]
        labels = [
            'Goals', 'xG', 'Assists', 'Key Passes',
            'Pass Acc%', 'Pass Acc F3rd%', '% Prog Passes',
            'Prog Carries', 'Threat Created',
            'Def Actions', 'Succ Def Actions',
            'Interceptions', 'Threat Prevented', 'Impact'
        ]
    elif position in ['LB', 'RB', 'LWB', 'RWB']:
        metrics = [
            'xA_per_90', 'assists_per_90', 'xG_per_90', 'goals_per_90',
            'successful_attacking_actions_per_90', 'pass_completion',
            'pass_completion_final_third', '%_passes_are_progressive',
            'prog_carries_per_90', 'total_threat_created_per_90',
            'successful_defensive_actions_per_90', 'tackle_win_rate',
            'interceptions_per_90', 'total_threat_prevented_per_90'
        ]
        labels = [
            'xA', 'Assists', 'xG', 'Goals',
            'Att Actions', 'Pass Acc%',
            'Pass Acc F3rd%', '% Prog Passes',
            'Prog Carries', 'Threat Created',
            'Succ Def Actions', 'Tackle%',
            'Interceptions', 'Threat Prevented'
        ]
    else:  # CB and other positions
        metrics = [
            'goals_per_90', 'keyPasses_per_90', 'xA_per_90',
            'prog_carries_per_90', 'successful_attacking_actions_per_90',
            'pass_completion', 'prog_passes_per_90', '%_passes_are_progressive',
            'passing_yards_per_90', 'clearances_per_90',
            'interceptions_per_90', 'tackle_win_rate',
            'def_aerial_win_rate', 'total_threat_prevented_per_90'
        ]
        labels = [
            'Goals', 'Key Passes', 'xA',
            'Prog Carries', 'Att Actions',
            'Pass Acc%', 'Prog Passes', '% Prog Passes',
            'Pass Yards', 'Clearances',
            'Interceptions', 'Tackle%',
            'Aerial%', 'Threat Prevented'
        ]
    
    try:
        # Вычисляем перцентили
        position_data = player_stats[player_stats['position_group'] == position].copy()
        
        logger.info(f"Position data shape: {position_data.shape}")
        
        if position_data.shape[0] < 2:
            logger.warning(f"Not enough players at position {position} for percentile calculation")
            return None
        
        def percentile_rank(series):
            valid = series.dropna()
            if len(valid) <= 1:
                return pd.Series([50] * len(series), index=series.index)
            ranks = valid.rank(method='average')
            pct = ((ranks - 1) / (len(valid) - 1)) * 100
            result = pd.Series([50] * len(series), index=series.index)
            result.loc[valid.index] = pct
            return result
        
        percentiles = {}
        for metric in metrics:
            if metric in position_data.columns:
                percentiles[metric] = percentile_rank(position_data[metric])
        
        # Получаем значения для игрока
        player_idx = player_data.index[0]
        values = [int(round(percentiles[m].loc[player_idx])) if m in percentiles else 50 for m in metrics]
        
        logger.info(f"Pizza values: {values}")
        
        # Создаем Pizza Chart
        baker = PyPizza(
            params=labels,
            straight_line_color="#000000",
            straight_line_lw=1.5,
            last_circle_lw=2,
            last_circle_color="#000000",
            other_circle_lw=1.5,
            other_circle_ls="-.",
            other_circle_color="#000000",
            inner_circle_size=20,
        )
        
        # Чередующиеся цвета
        slice_colors = [color1 if i % 2 == 0 else color2 for i in range(len(labels))]
        
        fig = plt.figure(figsize=(10, 8.5))
        fig.set_facecolor('white')
        ax = fig.add_axes([0.33, 0.13, 0.65, 0.7], projection='polar')
        
        baker.make_pizza(
            values,
            ax=ax,
            param_location=112,
            slice_colors=slice_colors,
            kwargs_slices=dict(
                edgecolor="#000000",
                zorder=1,
                linewidth=1
            ),
            kwargs_params=dict(
                color="#000000",
                fontsize=11,
                va="center",
                zorder=5
            ),
            kwargs_values=dict(
                color="#000000",
                fontsize=11,
                zorder=5,
                bbox=dict(
                    edgecolor="#000000",
                    facecolor="white",
                    boxstyle="round,pad=0.2",
                    lw=1
                )
            )
        )
        
        # Заголовок
        fig.text(0.645, 0.97, f"{player_name} - {team_name}", ha="center", size=18, color="#000000")
        fig.text(0.645, 0.942, f"Percentile Rank as {position}", ha="center", size=15, color="#000000")
        fig.text(0.99, 0.005, "Data: WTanalysis | Created: Novvi", size=9, color="#000000", ha="right")
        
        # Сохраняем
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        logger.info("Pizza chart created successfully")
        return buf
        
    except Exception as e:
        logger.error(f"Error creating pizza chart: {e}", exc_info=True)
        return None


def create_xt_grid_map(matchdata, playername, team_choice, position):
    """
    xT Grid Map - показывает где игрок создает угрозу
    """
    from mplsoccer import Pitch
    from matplotlib.colors import LinearSegmentedColormap
    
    # Filter player data
    player_data = filter_player_data(matchdata, playername, team_choice, position)
    player_data = player_data.loc[
        (player_data['typeId'].isin(['Pass', 'Carry'])) &
        (player_data['xT_value'].notna()) &
        (player_data['xT_value'] > 0)
    ].copy()
    
    if player_data.empty:
        return None
    
    player_data['x'] = pd.to_numeric(player_data['x'], errors='coerce')
    player_data['y'] = pd.to_numeric(player_data['y'], errors='coerce')
    player_data['end_x'] = pd.to_numeric(player_data['end_x'], errors='coerce')
    player_data['end_y'] = pd.to_numeric(player_data['end_y'], errors='coerce')
    player_data['xT_value'] = pd.to_numeric(player_data['xT_value'], errors='coerce')
    
    player_data = player_data.dropna(subset=['x', 'y', 'xT_value'])
    
    if len(player_data) < 5:
        return None
    
    # Create grid (24x16)
    x_bins = np.linspace(0, 100, 25)
    y_bins = np.linspace(0, 100, 17)
    
    xt_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    for _, row in player_data.iterrows():
        x_idx = np.digitize(row['x'], x_bins) - 1
        y_idx = np.digitize(row['y'], y_bins) - 1
        
        if 0 <= x_idx < len(x_bins)-1 and 0 <= y_idx < len(y_bins)-1:
            xt_grid[y_idx, x_idx] += row['xT_value']
    
    pitch = Pitch(
        pitch_type='opta',
        line_zorder=2,
        pitch_color='#FFFFFF',
        line_color='#000000',
        linewidth=1
    )
    
    fig, ax = pitch.draw(figsize=(18, 12))
    fig.set_facecolor('#FFFFFF')
    
    colors = ['#FFFFFF', '#FFB3BA', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('xt_cmap', colors, N=256)
    
    max_xt = xt_grid.max()
    if max_xt > 0:
        normalized_grid = xt_grid / max_xt
    else:
        normalized_grid = xt_grid
    
    for i in range(len(y_bins)-1):
        for j in range(len(x_bins)-1):
            x_start = x_bins[j]
            x_end = x_bins[j+1]
            y_start = y_bins[i]
            y_end = y_bins[i+1]
            
            color = cmap(normalized_grid[i, j])
            
            rect = plt.Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                facecolor=color,
                edgecolor='#CCCCCC',
                linewidth=0.5,
                zorder=1
            )
            ax.add_patch(rect)
            
            x_center = (x_start + x_end) / 2
            y_center = (y_start + y_end) / 2
            
            ax.text(
                x_center, y_center,
                f'{xt_grid[i, j]:.3f}',
                ha='center', va='center',
                fontsize=7,
                color='#000000',
                zorder=3
            )
    
    # Draw arrows for high xT actions
    high_xt_actions = player_data[player_data['xT_value'] > player_data['xT_value'].quantile(0.75)]
    
    for _, row in high_xt_actions.iterrows():
        if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
            ax.annotate('',
                       xy=(row['end_x'], row['end_y']),
                       xytext=(row['x'], row['y']),
                       arrowprops=dict(
                           arrowstyle='->',
                           color='#000000',
                           lw=1.5,
                           alpha=0.6,
                           zorder=2
                       ))
    
    ax.set_title(
        f"{playername} - xT Grid Map ({format_position_display(position)})",
        fontsize=14,
        pad=10,
        color="#000000",
        weight='bold'
    )
    
    ax.text(50, -3, 'Created: Novvi | Data: WTanalysis', 
            ha='center', fontsize=10, color='#000000', alpha=0.8)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#FFFFFF')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_creating_actions_map(matchdata, playername, team_choice, position):
    """
    Creating Actions - ТОЧНАЯ копия из Streamlit app.py
    Показывает Progressive Actions и Shot Assists
    """
    from mplsoccer import VerticalPitch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    
    def add_comet(ax, x0, y0, x1, y1, color, n=20, lw_start=0.6, lw_end=2.0, alpha_start=0.10, alpha_end=1.0, z=3):
        """Helper function to draw comet lines"""
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        
        segments = np.stack(
            (np.column_stack([xs[:-1], ys[:-1]]),
             np.column_stack([xs[1:], ys[1:]])),
            axis=1
        )
        
        widths = np.linspace(lw_start, lw_end, n - 1)
        alphas = np.linspace(alpha_start, alpha_end, n - 1)
        
        r, g, b, _ = to_rgba(color, 1.0)
        colors = [(r, g, b, a) for a in alphas]
        
        lc = LineCollection(
            segments,
            linewidths=widths,
            colors=colors,
            capstyle='round',
            joinstyle='round',
            zorder=z
        )
        ax.add_collection(lc)
    
    # Filter player data
    player_data = filter_player_data(matchdata, playername, team_choice, position)
    
    # Progressive actions
    progdata = player_data[
        (player_data.get('progressive_pass', 'No') == 'Yes') |
        (player_data.get('progressive_carry', 'No') == 'Yes')
    ].copy()
    
    # Shot assists
    shotassistdata = player_data[
        (player_data.get('keyPass', 0) == 1) |
        (player_data.get('assist', 0) == 1)
    ].copy()
    
    if progdata.empty and shotassistdata.empty:
        return None
    
    # Create figure - same layout as Player Actions
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.25), facecolor="#FAF9F6")
    plt.subplots_adjust(wspace=.1)
    
    pitch = VerticalPitch(
        pitch_type='opta',
        pitch_color="#FFFFFF",
        line_color="#000000",
        linewidth=2.5
    )
    
    # Draw all pitches
    pitch.draw(ax=axes[0])
    pitch.draw(ax=axes[1])
    pitch.draw(ax=axes[2])
    
    # PITCH 1 - Progressive Actions
    axes[0].set_title(f"{playername} - Progressive Actions as {position}", fontsize=10, color="#000000", weight='bold', pad=8)
    
    for _, row in progdata.iterrows():
        if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]
            
            if row.get("progressive_pass") == "Yes":
                add_comet(axes[0], x0, y0, x1, y1, color="#FF0000")
            
            if row.get("progressive_carry") == "Yes":
                add_comet(axes[0], x0, y0, x1, y1, color="#0000FF")
    
    axes[0].text(50, -5, "Red = Progressive Pass | Blue = Progressive Carry",
                 ha='center', fontsize=9, color="#000000")
    
    # PITCH 2 - Shot Assists
    axes[1].set_title(f"{playername} - Shot Assists as {position}", fontsize=10, color="#000000", weight='bold', pad=8)
    
    for _, row in shotassistdata.iterrows():
        if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]
            
            if row.get("keyPass", 0) == 1:
                add_comet(axes[1], x0, y0, x1, y1, color="#8B00FF")
            
            if row.get("assist", 0) == 1:
                add_comet(axes[1], x0, y0, x1, y1, color="#FF0000")
    
    axes[1].text(50, -5, "Purple = Shot Assist | Red = Assist",
                 ha='center', fontsize=9, color="#000000")
    
    # PITCH 3 - Empty for now (можно добавить что-то еще)
    axes[2].set_title(f"{playername} - All Creative Actions as {position}", fontsize=10, color="#000000", weight='bold', pad=8)
    
    # Combine all creative actions on pitch 3
    for _, row in progdata.iterrows():
        if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]
            add_comet(axes[2], x0, y0, x1, y1, color="#00FF00", alpha_start=0.05, alpha_end=0.6)
    
    for _, row in shotassistdata.iterrows():
        if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]
            add_comet(axes[2], x0, y0, x1, y1, color="#FF00FF", alpha_start=0.05, alpha_end=0.6)
    
    axes[2].text(50, -5, "Green = Progressive | Magenta = Shot Assist/Assist",
                 ha='center', fontsize=9, color="#000000")
    
    # Add player/team info
    axes[0].text(60, -13, f"{playername} - {team_choice}",
                 ha='left', fontsize=12, color="#000000", fontweight='bold')
    
    axes[2].text(1, -22, 'Data: WTanalysis | Created: Novvi', ha='right', fontsize=10, color="#000000")
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#FAF9F6')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_pass_network_viz(matchdata, playername, team_choice, position):
    """
    Pass Network - ТОЧНАЯ копия из Streamlit app.py
    Показывает пасы всей команды с выделением выбранного игрока
    """
    from mplsoccer import Pitch
    import matplotlib.patheffects as path_effects
    
    min_passes = 3
    
    # Get all team passes
    team_passes = matchdata.loc[
        (matchdata['team_name'] == team_choice) &
        (matchdata['typeId'] == 'Pass') &
        (matchdata['outcome'] == 'Successful')
    ].copy()
    
    if team_passes.empty or 'pass_recipient' not in team_passes.columns:
        return None
    
    # Remove passes with missing recipient
    team_passes = team_passes[team_passes['pass_recipient'].notna()]
    team_passes = team_passes[team_passes['pass_recipient'] != '']
    
    # Get all team players
    team_players = matchdata[matchdata['team_name'] == team_choice].copy()
    
    # Calculate average positions for all players
    average_locs = team_players.groupby('playerName').agg({'x': 'mean', 'y': 'mean'})
    
    # Count passes between players
    passes_between = team_passes.groupby(['playerName', 'pass_recipient']).size().reset_index(name='pass_count')
    passes_between = passes_between[passes_between['pass_count'] >= min_passes]
    
    if passes_between.empty:
        return None
    
    # Only keep passes where both players exist
    passes_between = passes_between[
        passes_between['playerName'].isin(average_locs.index) &
        passes_between['pass_recipient'].isin(average_locs.index)
    ]
    
    if passes_between.empty:
        return None
    
    # Merge with average locations
    passes_between = passes_between.merge(average_locs, left_on='playerName', right_index=True)
    passes_between = passes_between.merge(average_locs, left_on='pass_recipient', right_index=True, 
                                          suffixes=['', '_end'])
    
    # Calculate marker sizes
    average_locs_and_count = average_locs.copy()
    average_locs_and_count['pass_count'] = 0
    
    for idx in average_locs_and_count.index:
        sent = passes_between[passes_between['playerName'] == idx]['pass_count'].sum()
        received = passes_between[passes_between['pass_recipient'] == idx]['pass_count'].sum()
        average_locs_and_count.loc[idx, 'pass_count'] = sent + received
    
    # Only keep players involved in passes
    players_involved = set(passes_between['playerName'].unique()) | set(passes_between['pass_recipient'].unique())
    average_locs_and_count = average_locs_and_count.loc[average_locs_and_count.index.isin(players_involved)]
    
    # Remove players with no name
    average_locs_and_count = average_locs_and_count[average_locs_and_count.index.notna()]
    average_locs_and_count = average_locs_and_count[average_locs_and_count.index != '']
    
    if average_locs_and_count.empty:
        return None
    
    max_count = average_locs_and_count['pass_count'].max()
    average_locs_and_count['marker_size'] = (average_locs_and_count['pass_count'] / max_count * 1000) + 200
    
    # Calculate line widths
    max_passes = passes_between['pass_count'].max()
    passes_between['width'] = (passes_between['pass_count'] / max_passes * 10) + 1
    
    # Use horizontal Pitch
    pitch = Pitch(
        pitch_type='opta',
        pitch_color='white',
        line_color='black',
        linewidth=1
    )
    
    fig, axs = pitch.grid(
        figheight=10,
        title_height=0.08,
        endnote_space=0,
        axis=False,
        title_space=0,
        grid_height=0.82,
        endnote_height=0.03
    )
    fig.set_facecolor('white')
    
    # Draw pass lines
    pitch.lines(
        passes_between.x, passes_between.y,
        passes_between.x_end, passes_between.y_end,
        lw=passes_between.width,
        color='#BF616A',
        zorder=1,
        ax=axs['pitch'],
        alpha=0.8
    )
    
    # Draw outer nodes
    pitch.scatter(
        average_locs_and_count.x,
        average_locs_and_count.y,
        s=average_locs_and_count.marker_size,
        color='#BF616A',
        edgecolors='black',
        linewidth=0.5,
        alpha=1,
        ax=axs['pitch'],
        zorder=2
    )
    
    # Draw inner nodes (white)
    pitch.scatter(
        average_locs_and_count.x,
        average_locs_and_count.y,
        s=average_locs_and_count.marker_size / 2,
        color='white',
        edgecolors='black',
        linewidth=0.5,
        alpha=1,
        ax=axs['pitch'],
        zorder=3
    )
    
    # Highlight selected player
    if playername in average_locs_and_count.index:
        player_row = average_locs_and_count.loc[playername]
        pitch.scatter(
            player_row.x,
            player_row.y,
            s=player_row.marker_size,
            color='#BF616A',
            edgecolors='black',
            linewidth=3,
            alpha=1,
            ax=axs['pitch'],
            zorder=4
        )
    
    # Add player names
    for index, row in average_locs_and_count.iterrows():
        if not index or str(index).strip() == '':
            continue
        name_parts = str(index).strip().split()
        if not name_parts:
            continue
        display_name = name_parts[-1]
        
        text = pitch.annotate(
            display_name,
            xy=(row.x, row.y),
            c='black',
            va='center',
            ha='center',
            size=15,
            weight='bold',
            ax=axs['pitch'],
            zorder=5
        )
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                               path_effects.Normal()])
    
    # Add endnote
    axs['endnote'].text(
        1, 1,
        'Data: WTanalysis | Created: Novvi',
        color='black',
        va='center',
        ha='right',
        fontsize=12
    )
    
    # Add title
    axs['title'].text(
        0.5, 0.7,
        f'{team_choice} - Pass Network',
        color='black',
        va='center',
        ha='center',
        fontsize=20,
        weight='bold'
    )
    
    axs['title'].text(
        0.5, 0.15,
        f'Highlighting {playername}',
        color='black',
        va='center',
        ha='center',
        fontsize=14
    )
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return buf


