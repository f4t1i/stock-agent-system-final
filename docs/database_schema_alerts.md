# Alerts & Watchlist Database Schema

## Overview

Two SQLite databases for alerts and watchlist management:
- `data/alerts.db` - Alert definitions and history
- `data/watchlist.db` - Watchlists and symbols

---

## Alerts Database (`alerts.db`)

### Table: `alerts`

Stores alert definitions.

| Column | Type | Description |
|--------|------|-------------|
| `alert_id` | TEXT PRIMARY KEY | Unique alert identifier (UUID) |
| `user_id` | TEXT NOT NULL | User identifier |
| `symbol` | TEXT NOT NULL | Stock symbol |
| `alert_type` | TEXT NOT NULL | Alert type (price_threshold, confidence_change, etc.) |
| `condition` | TEXT NOT NULL | Condition (above, below, crosses_above, crosses_below, equals) |
| `threshold` | REAL NOT NULL | Threshold value |
| `notification_channels` | TEXT NOT NULL | JSON array of channels (email, push, webhook) |
| `enabled` | INTEGER NOT NULL | 1 if enabled, 0 if disabled |
| `last_triggered` | TEXT | ISO timestamp of last trigger |
| `created_at` | TEXT NOT NULL | ISO timestamp of creation |
| `metadata` | TEXT | JSON metadata |

**Indexes:**
- `idx_alerts_user` on `user_id`
- `idx_alerts_symbol` on `symbol`

**Example:**
```sql
INSERT INTO alerts VALUES (
  'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
  'user123',
  'AAPL',
  'price_threshold',
  'crosses_above',
  150.00,
  '["push", "email"]',
  1,
  NULL,
  '2026-01-05T10:00:00',
  '{}'
);
```

---

### Table: `alert_history`

Stores alert trigger history.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Auto-increment ID |
| `alert_id` | TEXT NOT NULL | Foreign key to alerts table |
| `triggered_at` | TEXT NOT NULL | ISO timestamp of trigger |
| `current_value` | REAL NOT NULL | Value that triggered alert |
| `notification_sent` | INTEGER NOT NULL | 1 if notification sent, 0 otherwise |

**Foreign Keys:**
- `alert_id` → `alerts(alert_id)`

**Example:**
```sql
INSERT INTO alert_history VALUES (
  1,
  'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
  '2026-01-05T14:30:00',
  151.50,
  1
);
```

---

## Watchlist Database (`watchlist.db`)

### Table: `watchlists`

Stores watchlist definitions.

| Column | Type | Description |
|--------|------|-------------|
| `watchlist_id` | TEXT PRIMARY KEY | Unique watchlist identifier (UUID) |
| `user_id` | TEXT NOT NULL | User identifier |
| `name` | TEXT NOT NULL | Watchlist name |
| `description` | TEXT | Watchlist description |
| `created_at` | TEXT NOT NULL | ISO timestamp of creation |

**Indexes:**
- `idx_watchlists_user` on `user_id`

**Example:**
```sql
INSERT INTO watchlists VALUES (
  'w1x2y3z4-a5b6-7890-cdef-123456789abc',
  'user123',
  'Tech Stocks',
  'My favorite technology stocks',
  '2026-01-05T10:00:00'
);
```

---

### Table: `watchlist_symbols`

Stores symbols in watchlists.

| Column | Type | Description |
|--------|------|-------------|
| `watchlist_id` | TEXT NOT NULL | Foreign key to watchlists table |
| `symbol` | TEXT NOT NULL | Stock symbol |
| `added_at` | TEXT NOT NULL | ISO timestamp when added |

**Primary Key:** `(watchlist_id, symbol)`

**Foreign Keys:**
- `watchlist_id` → `watchlists(watchlist_id)`

**Example:**
```sql
INSERT INTO watchlist_symbols VALUES (
  'w1x2y3z4-a5b6-7890-cdef-123456789abc',
  'AAPL',
  '2026-01-05T10:05:00'
);
```

---

## Queries

### Get all alerts for a user
```sql
SELECT * FROM alerts 
WHERE user_id = 'user123' 
ORDER BY created_at DESC;
```

### Get enabled alerts for a symbol
```sql
SELECT * FROM alerts 
WHERE symbol = 'AAPL' AND enabled = 1;
```

### Get alert trigger history
```sql
SELECT 
  a.symbol,
  a.alert_type,
  a.threshold,
  h.triggered_at,
  h.current_value,
  h.notification_sent
FROM alerts a
JOIN alert_history h ON a.alert_id = h.alert_id
WHERE a.user_id = 'user123'
ORDER BY h.triggered_at DESC
LIMIT 50;
```

### Get watchlist with symbol count
```sql
SELECT 
  w.*,
  COUNT(ws.symbol) as symbol_count
FROM watchlists w
LEFT JOIN watchlist_symbols ws ON w.watchlist_id = ws.watchlist_id
WHERE w.user_id = 'user123'
GROUP BY w.watchlist_id
ORDER BY w.created_at DESC;
```

### Get all symbols in a watchlist
```sql
SELECT symbol, added_at
FROM watchlist_symbols
WHERE watchlist_id = 'w1x2y3z4-a5b6-7890-cdef-123456789abc'
ORDER BY added_at DESC;
```

---

## Maintenance

### Cleanup old alert history
```sql
DELETE FROM alert_history
WHERE triggered_at < datetime('now', '-90 days');
```

### Delete disabled alerts older than 30 days
```sql
DELETE FROM alerts
WHERE enabled = 0 
AND created_at < datetime('now', '-30 days');
```

### Vacuum databases (reclaim space)
```sql
VACUUM;
```

---

## Backup

Recommended backup strategy:
```bash
# Backup alerts database
sqlite3 data/alerts.db ".backup data/alerts_backup.db"

# Backup watchlist database
sqlite3 data/watchlist.db ".backup data/watchlist_backup.db"
```

---

## Migration

When schema changes are needed:
1. Create migration script in `migrations/`
2. Apply with `sqlite3 data/alerts.db < migrations/001_add_column.sql`
3. Update this documentation
