# -*- coding: utf-8 -*-
import os
import io
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import pandas as pd
import numpy as np
import requests

# Импортируем функции визуализации
from viz_functions import (
    create_shot_map, 
    create_pass_and_carry_sonar,
    create_player_actions,
    create_zone_dominance,
    create_heatmap,
    create_pizza_chart,
    create_xt_grid_map,
    create_creating_actions_map,
    create_pass_network_viz,
    normalize_position,
    filter_player_data,
    format_position_display
)

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация GitHub
REPO_OWNER = "WTAnalysis"
REPO_NAME = "dashboard"
BRANCH = "main"

# Маппинг лиг
LEAGUES = {
    "�🇧 England - Premier League": "ENG1",
    "�🇧 England - Championship": "ENG2",
    "�🇧 England - League One": "ENG3",
    "�🇧 England - League Two": "ENG4",
    "🇪🇸 Spain - La Liga": "SPA1",
    "🇩🇪 Germany - Bundesliga": "GER1",
    "🇮🇹 Italy - Serie A": "ITA1",
    "🇫🇷 France - Ligue 1": "FRA1",
    "🇳🇱 Netherlands - Eredivisie": "NED1",
    "🇵🇹 Portugal - Primeira Liga": "POR1",
    "🇧🇪 Belgium - Pro League": "BEL1",
    "🇹🇷 Turkey - Super Lig": "TUR1",
    "🇬🇷 Greece - Super League": "GRE1",
    "🇦🇹 Austria - Bundesliga": "AUT1",
    "🇩🇰 Denmark - Superliga": "DEN1",
    "�🇧 Scotland - Premiership": "SCO1",
    "🇸🇦 Saudi Arabia - Pro League": "SAU1",
    "🇺🇸 USA - MLS": "USA1",
    "🇧🇷 Brazil - Brasileirao": "BRA1",
    "🇯🇵 Japan - J1 League": "JAP1",
    "🇮🇪 Ireland - Premier Division": "IRE1",
    "🇸🇪 Sweden - Allsvenskan": "SWE1",
    "🇦🇺 Australia - A-League": "AUS1",
}

# Маппинг сезонов для лиг (некоторые используют 2025 вместо 2526)
LEAGUE_SEASONS = {
    "USA1": "2025",  # MLS
    "BRA1": "2025",  # Brasileirão
    "JAP1": "2025",  # J1 League
    "IRE1": "2025",  # Premier Division
    "SWE1": "2025",  # Allsvenskan
}

DEFAULT_SEASON = "2526"

# Кэш данных
data_cache = {}

def build_github_url(filename):
    """Построить URL для файла на GitHub"""
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/{BRANCH}/{filename}"

def get_most_played_position(player_stats, player_name, team_name):
    """Получить позицию с наибольшим количеством сыгранных минут"""
    player_data = player_stats[
        (player_stats['player_name'] == player_name) &
        (player_stats['team_name'] == team_name)
    ]
    
    if player_data.empty:
        logger.warning(f"No data found for {player_name} at {team_name}")
        return None
    
    player_data_sorted = player_data.sort_values('minutes_played', ascending=False)
    top_position = player_data_sorted.iloc[0]['position_group']
    top_minutes = player_data_sorted.iloc[0]['minutes_played']
    
    top_position_normalized = normalize_position(top_position)
    
    logger.info(f"{player_name} - Top position: {top_position} (normalized: {top_position_normalized}) with {top_minutes} minutes")
    
    return top_position_normalized

def load_league_data(league_code):
    """Загрузить данные лиги с GitHub"""
    # Определяем сезон для лиги
    season = LEAGUE_SEASONS.get(league_code, DEFAULT_SEASON)
    cache_key = f"{league_code}_{season}"
    
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    try:
        parquet_url = build_github_url(f"{league_code}_{season}.parquet")
        logger.info(f"Loading parquet from: {parquet_url}")
        
        # Увеличенный timeout для больших файлов (MLS)
        matchdata = pd.read_parquet(parquet_url, storage_options={'timeout': 120})
        
        if "playing_position" in matchdata.columns:
            matchdata["playing_position"] = matchdata["playing_position"].apply(normalize_position)
        
        if "pass_recipient_position" in matchdata.columns:
            matchdata["pass_recipient_position"] = matchdata["pass_recipient_position"].apply(normalize_position)
        
        excel_url = build_github_url(f"{league_code}_{season}_playerstats_by_position_group.xlsx")
        logger.info(f"Loading excel from: {excel_url}")
        response = requests.get(excel_url, timeout=120)  # Увеличен timeout до 120 секунд
        response.raise_for_status()
        player_stats = pd.read_excel(io.BytesIO(response.content))
        
        data_cache[cache_key] = {
            'matchdata': matchdata,
            'player_stats': player_stats
        }
        
        logger.info(f"Successfully loaded data for {league_code}_{season}")
        return data_cache[cache_key]
    except requests.exceptions.Timeout:
        logger.error(f"Timeout loading data for {league_code}_{season} - file too large")
        return None
    except Exception as e:
        logger.error(f"Data loading error for {league_code}_{season}: {type(e).__name__}: {e}")
        return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    keyboard = []
    for league_name in LEAGUES.keys():
        keyboard.append([InlineKeyboardButton(league_name, callback_data=f"league_{league_name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Football Stats Bot\n\nChoose league:",
        reply_markup=reply_markup
    )

async def league_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора лиги"""
    query = update.callback_query
    await query.answer()
    
    league_name = query.data.replace("league_", "")
    league_code = LEAGUES[league_name]
    
    context.user_data['league_name'] = league_name
    context.user_data['league_code'] = league_code
    
    await query.edit_message_text(f"⏳ Loading {league_name}...")
    
    data = load_league_data(league_code)
    
    if data is None:
        # Определяем сезон для лиги
        season = LEAGUE_SEASONS.get(league_code, DEFAULT_SEASON)
        await query.edit_message_text(
            f"❌ Data loading error for {league_name}\n\n"
            f"League code: {league_code}\n"
            f"Season: {season}\n\n"
            f"This league may not be available yet. Try another league.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="back_to_leagues")]])
        )
        return
    
    teams = sorted(data['player_stats']['team_name'].unique())
    context.user_data['teams'] = teams
    
    keyboard = []
    for i in range(0, len(teams), 2):
        row = []
        row.append(InlineKeyboardButton(teams[i], callback_data=f"team_{i}"))
        if i + 1 < len(teams):
            row.append(InlineKeyboardButton(teams[i+1], callback_data=f"team_{i+1}"))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("Back", callback_data="back_to_leagues")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(f"{league_name}\n\nChoose team:", reply_markup=reply_markup)

async def team_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора команды"""
    query = update.callback_query
    await query.answer()
    
    team_idx = int(query.data.replace("team_", ""))
    team_name = context.user_data['teams'][team_idx]
    context.user_data['team_name'] = team_name
    
    league_code = context.user_data['league_code']
    data = load_league_data(league_code)
    
    players = data['player_stats'][
        data['player_stats']['team_name'] == team_name
    ]['player_name'].unique()
    players = sorted(players)
    context.user_data['players'] = players
    
    keyboard = []
    for i in range(0, len(players), 2):
        row = []
        row.append(InlineKeyboardButton(players[i][:20], callback_data=f"player_{i}"))
        if i + 1 < len(players):
            row.append(InlineKeyboardButton(players[i+1][:20], callback_data=f"player_{i+1}"))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("Back", callback_data=f"league_{context.user_data['league_name']}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(f"{team_name}\n\nChoose player:", reply_markup=reply_markup)

async def player_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора игрока"""
    query = update.callback_query
    await query.answer()
    
    player_idx = int(query.data.replace("player_", ""))
    player_name = context.user_data['players'][player_idx]
    context.user_data['player_name'] = player_name
    
    keyboard = [
        [InlineKeyboardButton("Statistics", callback_data="viz_stats")],
        [InlineKeyboardButton("Shot Map", callback_data="viz_shots")],
        [InlineKeyboardButton("Sonar", callback_data="viz_sonar")],
        [InlineKeyboardButton("Player Actions", callback_data="viz_actions")],
        [InlineKeyboardButton("Creating Actions", callback_data="viz_creating")],
        [InlineKeyboardButton("Heatmap", callback_data="viz_heatmap")],
        [InlineKeyboardButton("Zone Dominance", callback_data="viz_zone")],
        [InlineKeyboardButton("xT Grid Map", callback_data="viz_xt_grid")],
        [InlineKeyboardButton("Pass Network", callback_data="viz_pass_network")],
        [InlineKeyboardButton("Pizza Chart", callback_data="viz_pizza")],
        [InlineKeyboardButton("Back", callback_data=f"team_{context.user_data['teams'].index(context.user_data['team_name'])}")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"{player_name}\n{context.user_data['league_name']}\n{context.user_data['team_name']}\n\nChoose visualization:",
        reply_markup=reply_markup
    )

async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать расширенную статистику - страница 1"""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("Loading...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    player_stats = data['player_stats']
    
    player_data = player_stats[
        (player_stats['player_name'] == player_name) &
        (player_stats['team_name'] == team_name)
    ]
    
    if player_data.empty:
        await query.edit_message_text("Data not found")
        return
    
    # Aggregate stats across all positions
    total_minutes = player_data['minutes_played'].sum()
    
    def weighted_avg(col):
        if col not in player_data.columns:
            return 0
        return (player_data[col] * player_data['minutes_played']).sum() / total_minutes if total_minutes > 0 else 0
    
    stats_text = f"📊 {player_name}\n{context.user_data['league_name']} | {team_name}\n\n"
    
    # BASIC STATS
    stats_text += "⚽ BASIC STATS\n"
    stats_text += f"Minutes: {total_minutes:.0f}\n"
    stats_text += f"Goals: {player_data['goals'].sum():.0f}\n"
    stats_text += f"xG: {player_data['xG'].sum():.2f}\n"
    stats_text += f"Assists: {player_data['assists'].sum():.0f}\n"
    stats_text += f"xA: {player_data['xA'].sum():.2f}\n\n"
    
    # SHOOTING
    stats_text += "🎯 SHOOTING\n"
    stats_text += f"Shots per 90: {weighted_avg('shots_per_90'):.2f}\n"
    stats_text += f"Shots on Target per 90: {weighted_avg('shots_on_target_per_90'):.2f}\n"
    stats_text += f"Shot Accuracy: {weighted_avg('shot_accuracy')*100:.1f}%\n"
    stats_text += f"xG per 90: {weighted_avg('xG_per_90'):.2f}\n"
    stats_text += f"xGOT per 90: {weighted_avg('xGOT_per_90'):.2f}\n"
    stats_text += f"Goals per 90: {weighted_avg('goals_per_90'):.2f}\n"
    stats_text += f"Shot Quality: {weighted_avg('shot_quality')*100:.1f}%\n\n"
    
    # PASSING
    stats_text += "🎯 PASSING\n"
    stats_text += f"Pass Completion: {weighted_avg('pass_completion')*100:.1f}%\n"
    stats_text += f"Pass Comp Final Third: {weighted_avg('pass_completion_final_third')*100:.1f}%\n"
    stats_text += f"Passes per 90: {weighted_avg('attempted_passes_per_90'):.2f}\n"
    stats_text += f"Progressive Passes per 90: {weighted_avg('prog_passes_per_90'):.2f}\n"
    stats_text += f"% Progressive Passes: {weighted_avg('%_passes_are_progressive')*100:.1f}%\n"
    stats_text += f"Key Passes per 90: {weighted_avg('keyPasses_per_90'):.2f}\n"
    stats_text += f"Assists per 90: {weighted_avg('assists_per_90'):.2f}\n"
    stats_text += f"xA per 90: {weighted_avg('xA_per_90'):.2f}\n"
    
    keyboard = [
        [InlineKeyboardButton("Next Page →", callback_data="stats_page2")],
        [InlineKeyboardButton("Back", callback_data=f"player_{context.user_data['players'].index(player_name)}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(stats_text, reply_markup=reply_markup)

async def show_stats_page2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать статистику - страница 2"""
    query = update.callback_query
    await query.answer()
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    player_stats = data['player_stats']
    
    player_data = player_stats[
        (player_stats['player_name'] == player_name) &
        (player_stats['team_name'] == team_name)
    ]
    
    total_minutes = player_data['minutes_played'].sum()
    
    def weighted_avg(col):
        if col not in player_data.columns:
            return 0
        return (player_data[col] * player_data['minutes_played']).sum() / total_minutes if total_minutes > 0 else 0
    
    stats_text = f"📊 {player_name} (Page 2/3)\n{context.user_data['league_name']} | {team_name}\n\n"
    
    # DRIBBLING & CARRYING
    stats_text += "🏃 DRIBBLING & CARRYING\n"
    stats_text += f"Dribbles per 90: {weighted_avg('dribbles_per_90'):.2f}\n"
    stats_text += f"Progressive Carries per 90: {weighted_avg('prog_carries_per_90'):.2f}\n"
    stats_text += f"Carries to Final Third per 90: {weighted_avg('carries_to_final_third_per_90'):.2f}\n"
    stats_text += f"Carrying Yards per 90: {weighted_avg('carrying_yards_per_90'):.2f}\n"
    stats_text += f"10+ Yard Carries per 90: {weighted_avg('ten_yard_carries_per_90'):.2f}\n"
    stats_text += f"Fouled per 90: {weighted_avg('fouled_per_90'):.2f}\n\n"
    
    # TOUCHES
    stats_text += "👆 TOUCHES\n"
    stats_text += f"Touches per 90: {weighted_avg('touches_per_90'):.2f}\n"
    stats_text += f"Touches Final Third per 90: {weighted_avg('touches_in_final_third_per_90'):.2f}\n"
    stats_text += f"Touches Middle Third per 90: {weighted_avg('touches_in_middle_third_per_90'):.2f}\n"
    stats_text += f"Touches Own Third per 90: {weighted_avg('touches_in_own_third_per_90'):.2f}\n"
    stats_text += f"Touches in Box per 90: {weighted_avg('touches_in_box_per_90'):.2f}\n"
    stats_text += f"Received Passes per 90: {weighted_avg('received_passes_per_90'):.2f}\n\n"
    
    # DEFENDING
    stats_text += "🛡️ DEFENDING\n"
    stats_text += f"Tackles per 90: {weighted_avg('tackles_per_90'):.2f}\n"
    stats_text += f"Tackle Win Rate: {weighted_avg('tackle_win_rate')*100:.1f}%\n"
    stats_text += f"Interceptions per 90: {weighted_avg('interceptions_per_90'):.2f}\n"
    stats_text += f"Ball Recoveries per 90: {weighted_avg('ball_recoveries_per_90'):.2f}\n"
    stats_text += f"Clearances per 90: {weighted_avg('clearances_per_90'):.2f}\n"
    stats_text += f"Blocked Shots per 90: {weighted_avg('blocked_shots_per_90'):.2f}\n"
    stats_text += f"Aerials per 90: {weighted_avg('aerials_per_90'):.2f}\n"
    stats_text += f"Aerial Win Rate: {weighted_avg('aerial_win_rate')*100:.1f}%\n"
    
    keyboard = [
        [InlineKeyboardButton("← Previous", callback_data="viz_stats"), InlineKeyboardButton("Next →", callback_data="stats_page3")],
        [InlineKeyboardButton("Back", callback_data=f"player_{context.user_data['players'].index(player_name)}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(stats_text, reply_markup=reply_markup)

async def show_stats_page3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать статистику - страница 3"""
    query = update.callback_query
    await query.answer()
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    player_stats = data['player_stats']
    
    player_data = player_stats[
        (player_stats['player_name'] == player_name) &
        (player_stats['team_name'] == team_name)
    ]
    
    total_minutes = player_data['minutes_played'].sum()
    
    def weighted_avg(col):
        if col not in player_data.columns:
            return 0
        return (player_data[col] * player_data['minutes_played']).sum() / total_minutes if total_minutes > 0 else 0
    
    stats_text = f"📊 {player_name} (Page 3/3)\n{context.user_data['league_name']} | {team_name}\n\n"
    
    # THREAT & IMPACT
    stats_text += "⚡ THREAT & IMPACT\n"
    stats_text += f"Threat Created per 90: {weighted_avg('total_threat_created_per_90'):.2f}\n"
    stats_text += f"Passing Threat per 90: {weighted_avg('passing_threat_per_90'):.2f}\n"
    stats_text += f"Carry Threat per 90: {weighted_avg('carry_threat_per_90'):.2f}\n"
    stats_text += f"Threat Prevented per 90: {weighted_avg('total_threat_prevented_per_90'):.2f}\n"
    stats_text += f"Player Impact per 90: {weighted_avg('threat_value_per_90'):.2f}\n\n"
    
    # ATTACKING ACTIONS
    stats_text += "⚔️ ATTACKING ACTIONS\n"
    stats_text += f"Attacking Actions per 90: {weighted_avg('attacking_actions_per_90'):.2f}\n"
    stats_text += f"Successful Att Actions per 90: {weighted_avg('successful_attacking_actions_per_90'):.2f}\n\n"
    
    # DEFENSIVE ACTIONS
    stats_text += "🛡️ DEFENSIVE ACTIONS\n"
    stats_text += f"Defensive Actions per 90: {weighted_avg('defensive_actions_per_90'):.2f}\n"
    stats_text += f"Successful Def Actions per 90: {weighted_avg('successful_defensive_actions_per_90'):.2f}\n\n"
    
    # DISCIPLINE
    stats_text += "🟨 DISCIPLINE\n"
    stats_text += f"Fouls per 90: {weighted_avg('fouls_per_90'):.2f}\n"
    stats_text += f"Yellow Cards per 90: {weighted_avg('yellow_cards_per_90'):.2f}\n"
    stats_text += f"Red Cards per 90: {weighted_avg('red_cards_per_90'):.2f}\n\n"
    
    # POSITIONS
    stats_text += "📍 POSITIONS\n"
    for _, row in player_data.sort_values('minutes_played', ascending=False).iterrows():
        pos = row.get('position_group', 'N/A')
        mins = row.get('minutes_played', 0)
        stats_text += f"{pos}: {mins:.0f} min\n"
    
    keyboard = [
        [InlineKeyboardButton("← Previous", callback_data="stats_page2")],
        [InlineKeyboardButton("Back", callback_data=f"player_{context.user_data['players'].index(player_name)}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(stats_text, reply_markup=reply_markup)

async def show_shot_map(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать карту ударов"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_shot_map(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough shot data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Shot Map as {position}",
        reply_markup=reply_markup
    )

async def show_sonar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать sonar"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_pass_and_carry_sonar(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Passing & Carrying Sonars as {position}",
        reply_markup=reply_markup
    )

async def show_player_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать player actions"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_player_actions(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Player Actions as {position}",
        reply_markup=reply_markup
    )

async def show_zone_dominance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать zone dominance"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_zone_dominance(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Zone Dominance as {position}",
        reply_markup=reply_markup
    )

async def show_heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать heatmap"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_heatmap(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Touch Heatmap as {position}",
        reply_markup=reply_markup
    )

async def show_creating_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать creating actions"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_creating_actions_map(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Creating Actions as {position}",
        reply_markup=reply_markup
    )

async def show_xt_grid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать xT grid map"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_xt_grid_map(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - xT Grid Map as {position}",
        reply_markup=reply_markup
    )

async def show_pass_network_viz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать pass network"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    data = load_league_data(league_code)
    matchdata = data['matchdata']
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    buf = create_pass_network_viz(matchdata, player_name, team_name, position)
    
    if buf is None:
        await query.message.reply_text("❌ Not enough data")
        return
    
    player_idx = context.user_data['players'].index(player_name)
    keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"{player_name} - Pass Network as {position}",
        reply_markup=reply_markup
    )

async def show_pizza_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать меню выбора цветов для pizza chart"""
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("Blue & Yellow", callback_data="pizza_blue_yellow")],
        [InlineKeyboardButton("Red & Green", callback_data="pizza_red_green")],
        [InlineKeyboardButton("Purple & Orange", callback_data="pizza_purple_orange")],
        [InlineKeyboardButton("Pink & Cyan", callback_data="pizza_pink_cyan")],
        [InlineKeyboardButton("Dark Blue & Gold", callback_data="pizza_darkblue_gold")],
        [InlineKeyboardButton("Green & Red", callback_data="pizza_green_red")],
        [InlineKeyboardButton("Back", callback_data=f"player_{context.user_data['players'].index(context.user_data['player_name'])}")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("Choose Pizza Chart color scheme:", reply_markup=reply_markup)

async def show_pizza_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать pizza chart"""
    query = update.callback_query
    await query.answer()
    
    await query.message.reply_text("⏳ иди нахуй, жди пару секунд...")
    
    player_name = context.user_data['player_name']
    team_name = context.user_data['team_name']
    league_code = context.user_data['league_code']
    
    logger.info(f"Pizza chart requested for {player_name} at {team_name}")
    
    data = load_league_data(league_code)
    player_stats = data['player_stats']
    
    position = get_most_played_position(player_stats, player_name, team_name)
    
    logger.info(f"Position for pizza chart: {position}")
    
    if position is None:
        await query.message.reply_text("❌ Position data not found")
        return
    
    # Определяем цвета
    color_scheme = query.data.replace("pizza_", "")
    color_schemes = {
        "blue_yellow": ("#1E88E5", "#FFC107"),
        "red_green": ("#E53935", "#43A047"),
        "purple_orange": ("#8E24AA", "#FB8C00"),
        "pink_cyan": ("#D81B60", "#00ACC1"),
        "darkblue_gold": ("#1565C0", "#FFD700"),
        "green_red": ("#2E7D32", "#C62828")
    }
    
    color1, color2 = color_schemes.get(color_scheme, ("#1E88E5", "#FFC107"))
    
    logger.info(f"Creating pizza chart with colors {color1}, {color2}")
    
    try:
        buf = create_pizza_chart(player_stats, player_name, team_name, position, color1, color2)
        
        if buf is None:
            logger.warning("Pizza chart returned None")
            await query.message.reply_text("❌ Not enough data for pizza chart")
            return
        
        logger.info("Pizza chart created successfully, sending photo")
        
        player_idx = context.user_data['players'].index(player_name)
        keyboard = [[InlineKeyboardButton("Back", callback_data=f"player_{player_idx}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_photo(
            photo=buf,
            caption=f"{player_name} - Pizza Chart as {position}",
            reply_markup=reply_markup
        )
        
        logger.info("Pizza chart sent successfully")
        
    except Exception as e:
        logger.error(f"Error in show_pizza_chart: {e}", exc_info=True)
        await query.message.reply_text(f"❌ Error creating pizza chart: {str(e)}")

async def back_to_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вернуться к выбору лиги"""
    query = update.callback_query
    await query.answer()
    
    keyboard = []
    for league_name in LEAGUES.keys():
        keyboard.append([InlineKeyboardButton(league_name, callback_data=f"league_{league_name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("Choose league:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Роутер для всех callback queries"""
    query = update.callback_query
    data = query.data
    
    if data.startswith("league_"):
        await league_selected(update, context)
    elif data.startswith("team_"):
        await team_selected(update, context)
    elif data.startswith("player_"):
        await player_selected(update, context)
    elif data == "viz_stats":
        await show_stats(update, context)
    elif data == "stats_page2":
        await show_stats_page2(update, context)
    elif data == "stats_page3":
        await show_stats_page3(update, context)
    elif data == "viz_shots":
        await show_shot_map(update, context)
    elif data == "viz_sonar":
        await show_sonar(update, context)
    elif data == "viz_actions":
        await show_player_actions(update, context)
    elif data == "viz_zone":
        await show_zone_dominance(update, context)
    elif data == "viz_heatmap":
        await show_heatmap(update, context)
    elif data == "viz_creating":
        await show_creating_actions(update, context)
    elif data == "viz_xt_grid":
        await show_xt_grid(update, context)
    elif data == "viz_pass_network":
        await show_pass_network_viz(update, context)
    elif data == "viz_pizza":
        await show_pizza_menu(update, context)
    elif data.startswith("pizza_"):
        await show_pizza_chart(update, context)
    elif data == "back_to_leagues":
        await back_to_leagues(update, context)

def main():
    """Запуск бота"""
    TOKEN = "8394835981:AAE1tyIhibirFF8nVLZziw5ToFkwjtc93rI"
    
    from telegram.request import HTTPXRequest
    
    request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0
    )
    
    application = Application.builder().token(TOKEN).request(request).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    
    logger.info("Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
