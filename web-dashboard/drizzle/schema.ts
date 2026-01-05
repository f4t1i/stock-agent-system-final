import { int, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Analysis History Table
 * Stores all stock analysis results from the 5 AI agents
 */
export const analysisHistory = mysqlTable("analysis_history", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  symbol: varchar("symbol", { length: 10 }).notNull(),
  analysisType: varchar("analysisType", { length: 50 }).notNull(), // 'full', 'quick', 'openbb'
  
  // Agent results
  newsAgentResult: text("newsAgentResult"),
  technicalAgentResult: text("technicalAgentResult"),
  fundamentalAgentResult: text("fundamentalAgentResult"),
  strategistAgentResult: text("strategistAgentResult"),
  supervisorResult: text("supervisorResult"),
  
  // Final recommendation
  recommendation: mysqlEnum("recommendation", ["buy", "sell", "hold"]),
  confidence: int("confidence"), // 0-100
  targetPrice: varchar("targetPrice", { length: 20 }),
  
  // Metadata
  executionTime: int("executionTime"), // milliseconds
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type AnalysisHistory = typeof analysisHistory.$inferSelect;
export type InsertAnalysisHistory = typeof analysisHistory.$inferInsert;

/**
 * Portfolio Holdings Table
 * Tracks user's stock positions and performance
 */
export const portfolioHoldings = mysqlTable("portfolio_holdings", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  symbol: varchar("symbol", { length: 10 }).notNull(),
  
  // Position details
  quantity: int("quantity").notNull(),
  averagePrice: varchar("averagePrice", { length: 20 }).notNull(),
  currentPrice: varchar("currentPrice", { length: 20 }),
  
  // Performance
  totalValue: varchar("totalValue", { length: 20 }),
  profitLoss: varchar("profitLoss", { length: 20 }),
  profitLossPercent: varchar("profitLossPercent", { length: 10 }),
  
  // Trade info
  entryDate: timestamp("entryDate").notNull(),
  exitDate: timestamp("exitDate"),
  status: mysqlEnum("status", ["active", "closed"]).default("active").notNull(),
  
  // Metadata
  notes: text("notes"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type PortfolioHolding = typeof portfolioHoldings.$inferSelect;
export type InsertPortfolioHolding = typeof portfolioHoldings.$inferInsert;

/**
 * Training Logs Table
 * Records AI agent training sessions and performance metrics
 */
export const trainingLogs = mysqlTable("training_logs", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  
  // Training session
  sessionId: varchar("sessionId", { length: 64 }).notNull(),
  agentName: varchar("agentName", { length: 50 }).notNull(), // 'news', 'technical', 'fundamental', 'strategist', 'supervisor'
  epoch: int("epoch").notNull(),
  
  // Metrics
  loss: varchar("loss", { length: 20 }),
  accuracy: varchar("accuracy", { length: 20 }),
  precision: varchar("precision", { length: 20 }),
  recall: varchar("recall", { length: 20 }),
  f1Score: varchar("f1Score", { length: 20 }),
  
  // Training config
  learningRate: varchar("learningRate", { length: 20 }),
  batchSize: int("batchSize"),
  modelVersion: varchar("modelVersion", { length: 50 }),
  
  // Status
  status: mysqlEnum("status", ["running", "completed", "failed"]).notNull(),
  duration: int("duration"), // seconds
  
  // Metadata
  notes: text("notes"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type TrainingLog = typeof trainingLogs.$inferSelect;
export type InsertTrainingLog = typeof trainingLogs.$inferInsert;