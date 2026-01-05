import { getSessionCookieOptions } from "./_core/cookies";
import { COOKIE_NAME } from "../shared/const";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import * as db from "./db";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,

  // Alerts Router
  alerts: router({    createAlert: protectedProcedure
      .input(z.object({
        symbol: z.string(),
        alert_type: z.enum(["price_threshold", "confidence_change", "recommendation_change", "technical_signal"]),
        condition: z.enum(["above", "below", "crosses_above", "crosses_below", "equals"]),
        threshold: z.number(),
        notification_channels: z.array(z.enum(["email", "push", "webhook"])),
      }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { alert_id: Math.random().toString(36).substring(7), ...input, enabled: true, created_at: new Date() };
      }),

    listAlerts: protectedProcedure
      .query(async () => {
        // TODO: Call Python backend API
        return [];
      }),

    toggleAlert: protectedProcedure
      .input(z.object({ alert_id: z.string(), enabled: z.boolean() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { success: true };
      }),

    deleteAlert: protectedProcedure
      .input(z.object({ alert_id: z.string() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { success: true };
      }),
  }),

  // Watchlist Router
  watchlist: router({
    createWatchlist: protectedProcedure
      .input(z.object({ name: z.string(), description: z.string().optional() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { watchlist_id: Math.random().toString(36).substring(7), ...input, symbol_count: 0, created_at: new Date() };
      }),

    listWatchlists: protectedProcedure
      .query(async () => {
        // TODO: Call Python backend API
        return [];
      }),

    addSymbol: protectedProcedure
      .input(z.object({ watchlist_id: z.string(), symbol: z.string() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { success: true };
      }),

    removeSymbol: protectedProcedure
      .input(z.object({ watchlist_id: z.string(), symbol: z.string() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { success: true };
      }),

    deleteWatchlist: protectedProcedure
      .input(z.object({ watchlist_id: z.string() }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        return { success: true };
      }),
  }),

  // Explainability Router
  explainability: router({
    getDecision: protectedProcedure
      .input(z.object({
        decisionId: z.string(),
      }))
      .query(async ({ input }) => {
        // TODO: Call Python backend API
        // For now, return mock data
        return {
          decisionId: input.decisionId,
          symbol: "AAPL",
          agentName: "news_agent",
          recommendation: "BUY" as const,
          confidence: 0.85,
          reasoning: "Strong positive sentiment from recent earnings report and product announcements. Market momentum is favorable.",
          keyFactors: [
            {
              name: "Sentiment Score",
              importance: 0.9,
              value: 1.5,
              description: "News sentiment: 1.50 (-2 to 2)",
            },
            {
              name: "Key Events",
              importance: 0.8,
              value: 3,
              description: "Significant events: Earnings beat, New product launch, CEO interview",
            },
            {
              name: "News Volume",
              importance: 0.6,
              value: 15,
              description: "Number of articles analyzed: 15",
            },
          ],
          alternatives: [
            {
              scenario: "Conservative approach",
              recommendation: "HOLD",
              confidence: 0.6,
              reasoning: "A wait-and-see approach to gather more information before acting.",
            },
          ],
          timestamp: new Date(),
          metadata: {},
        };
      }),

    analyze: protectedProcedure
      .input(z.object({
        symbol: z.string(),
        agentName: z.string(),
        agentOutput: z.record(z.any()),
        context: z.record(z.any()).optional(),
      }))
      .mutation(async ({ input }) => {
        // TODO: Call Python backend API
        // For now, return mock data
        const decisionId = Math.random().toString(36).substring(7);
        return {
          decisionId,
          symbol: input.symbol,
          agentName: input.agentName,
          recommendation: "BUY" as const,
          confidence: 0.75,
          reasoning: "Analysis based on provided agent output.",
          keyFactors: [],
          alternatives: [],
          timestamp: new Date(),
          metadata: input.context || {},
        };
      }),

    listRecent: protectedProcedure
      .input(z.object({
        limit: z.number().optional(),
        agentName: z.string().optional(),
        symbol: z.string().optional(),
      }).optional())
      .query(async ({ input }) => {
        // TODO: Call Python backend API
        // For now, return mock data
        return [
          {
            decisionId: "dec1",
            symbol: "AAPL",
            agentName: "news_agent",
            recommendation: "BUY" as const,
            confidence: 0.85,
            timestamp: new Date(),
          },
          {
            decisionId: "dec2",
            symbol: "GOOGL",
            agentName: "technical_agent",
            recommendation: "HOLD" as const,
            confidence: 0.65,
            timestamp: new Date(Date.now() - 3600000),
          },
        ];
      }),
  }),
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // Analysis History Router
  analysis: router({
    create: protectedProcedure
      .input(z.object({
        symbol: z.string(),
        analysisType: z.string(),
        newsAgentResult: z.string().optional(),
        technicalAgentResult: z.string().optional(),
        fundamentalAgentResult: z.string().optional(),
        strategistAgentResult: z.string().optional(),
        supervisorResult: z.string().optional(),
        recommendation: z.enum(["buy", "sell", "hold"]).optional(),
        confidence: z.number().optional(),
        targetPrice: z.string().optional(),
        executionTime: z.number().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        return await db.createAnalysis({
          userId: ctx.user.id,
          ...input,
        });
      }),
    
    list: protectedProcedure
      .input(z.object({
        limit: z.number().optional(),
      }).optional())
      .query(async ({ ctx, input }) => {
        return await db.getAnalysisByUser(ctx.user.id, input?.limit);
      }),
    
    bySymbol: protectedProcedure
      .input(z.object({
        symbol: z.string(),
      }))
      .query(async ({ ctx, input }) => {
        return await db.getAnalysisBySymbol(ctx.user.id, input.symbol);
      }),
  }),

  // Portfolio Router
  portfolio: router({
    create: protectedProcedure
      .input(z.object({
        symbol: z.string(),
        quantity: z.number(),
        averagePrice: z.string(),
        currentPrice: z.string().optional(),
        entryDate: z.date(),
        notes: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        return await db.createPortfolioHolding({
          userId: ctx.user.id,
          status: "active",
          ...input,
        });
      }),
    
    list: protectedProcedure
      .query(async ({ ctx }) => {
        return await db.getPortfolioByUser(ctx.user.id);
      }),
    
    update: protectedProcedure
      .input(z.object({
        id: z.number(),
        currentPrice: z.string().optional(),
        totalValue: z.string().optional(),
        profitLoss: z.string().optional(),
        profitLossPercent: z.string().optional(),
        notes: z.string().optional(),
      }))
      .mutation(async ({ input }) => {
        const { id, ...updates } = input;
        return await db.updatePortfolioHolding(id, updates);
      }),
    
    close: protectedProcedure
      .input(z.object({
        id: z.number(),
      }))
      .mutation(async ({ input }) => {
        return await db.closePortfolioPosition(input.id);
      }),
  }),

  // Training Logs Router
  training: router({
    create: protectedProcedure
      .input(z.object({
        sessionId: z.string(),
        agentName: z.string(),
        epoch: z.number(),
        loss: z.string().optional(),
        accuracy: z.string().optional(),
        precision: z.string().optional(),
        recall: z.string().optional(),
        f1Score: z.string().optional(),
        learningRate: z.string().optional(),
        batchSize: z.number().optional(),
        modelVersion: z.string().optional(),
        status: z.enum(["running", "completed", "failed"]),
        duration: z.number().optional(),
        notes: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        return await db.createTrainingLog({
          userId: ctx.user.id,
          ...input,
        });
      }),
    
    list: protectedProcedure
      .input(z.object({
        limit: z.number().optional(),
      }).optional())
      .query(async ({ ctx, input }) => {
        return await db.getTrainingLogsByUser(ctx.user.id, input?.limit);
      }),
    
    bySession: protectedProcedure
      .input(z.object({
        sessionId: z.string(),
      }))
      .query(async ({ input }) => {
        return await db.getTrainingLogsBySession(input.sessionId);
      }),
    
    byAgent: protectedProcedure
      .input(z.object({
        agentName: z.string(),
      }))
      .query(async ({ ctx, input }) => {
        return await db.getTrainingLogsByAgent(ctx.user.id, input.agentName);
      }),
  }),
});

export type AppRouter = typeof appRouter;
