import { getSessionCookieOptions } from "./_core/cookies";
import { COOKIE_NAME } from "../shared/const";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import * as db from "./db";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
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
