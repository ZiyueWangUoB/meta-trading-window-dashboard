"""Project-wide constants and configuration."""

from __future__ import annotations

from pathlib import Path

TICKER = "META"
COMPANY_NAME = "Meta Platforms, Inc."
CIK = "0001326801"

# SEC requires a descriptive User-Agent with contact details on automated requests.
SEC_USER_AGENT = "meta-trading-window-dashboard/1.0 (wangziyue@hotmail.co.uk)"

# Meta's closed trading period starts at market close on the last day of the
# middle month of each quarter (Feb, May, Aug, Nov) and ends when the window
# reopens on the second full trading day after quarterly earnings.
MIDDLE_MONTHS = (2, 5, 8, 11)
MIDDLE_MONTH_NAMES = {2: "Feb", 5: "May", 8: "Aug", 11: "Nov"}
WINDOW_OPEN_TRADING_DAY_OFFSET = 2

# History: META listed on 2012-05-18; the first post-IPO earnings 8-K (Q2 2012)
# was filed 2012-07-26, giving the first open window in August 2012.
PRICE_HISTORY_START = "2012-05-18"
EARNINGS_HISTORY_START = "2012-01-01"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

EARNINGS_CACHE_TTL_DAYS = 7
PRICE_CACHE_TTL_HOURS = 12
