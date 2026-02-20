# Football Stats Telegram Bot - Deployment Guide

## Deploy to Render.com (Free)

### Step 1: Push to GitHub

1. Open terminal in `telegram_bot` folder
2. Initialize git repository:
```bash
git init
git add .
git commit -m "Initial commit - Football Stats Bot"
```

3. Create new repository on GitHub:
   - Go to https://github.com/new
   - Name: `football-stats-telegram-bot`
   - Make it Public or Private
   - Don't initialize with README (we already have files)
   - Click "Create repository"

4. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/football-stats-telegram-bot.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render.com

1. Go to https://render.com and sign up (use GitHub account)

2. Click "New +" → "Background Worker"

3. Connect your GitHub repository:
   - Select `football-stats-telegram-bot`
   - Click "Connect"

4. Configure the service:
   - **Name**: `football-stats-bot`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`

5. Click "Create Background Worker"

6. Wait for deployment (2-3 minutes)

7. Check logs to see if bot is running

### Step 3: Verify Bot is Working

1. Open Telegram
2. Send `/start` to your bot
3. Try selecting a league and player

Done! Your bot is now running 24/7 on Render.com for free!

## Troubleshooting

If bot doesn't work:
- Check Render logs for errors
- Make sure all files are pushed to GitHub
- Verify bot token is correct in bot.py

## Updating the Bot

When you make changes:
```bash
git add .
git commit -m "Update bot"
git push
```

Render will automatically redeploy!
