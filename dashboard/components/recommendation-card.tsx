"use client"

import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, Shield, Target, DollarSign, Percent } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import numeral from "numeral"

interface RecommendationCardProps {
  recommendation: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
  entry: number
  stopLoss: number
  takeProfit: number
  positionSize: number // as percentage
  confidence: number // 0 to 1
  reasoning: string
}

export function RecommendationCard({
  recommendation,
  entry,
  stopLoss,
  takeProfit,
  positionSize,
  confidence,
  reasoning,
}: RecommendationCardProps) {
  const getRecommendationConfig = () => {
    switch (recommendation) {
      case "strong_buy":
        return {
          label: "STRONG BUY",
          color: "text-green-500",
          bg: "bg-green-500/10",
          border: "border-green-500",
          icon: TrendingUp,
          gradient: "from-green-500/20 to-transparent",
        }
      case "buy":
        return {
          label: "BUY",
          color: "text-green-400",
          bg: "bg-green-400/10",
          border: "border-green-400",
          icon: TrendingUp,
          gradient: "from-green-400/20 to-transparent",
        }
      case "sell":
        return {
          label: "SELL",
          color: "text-red-400",
          bg: "bg-red-400/10",
          border: "border-red-400",
          icon: TrendingDown,
          gradient: "from-red-400/20 to-transparent",
        }
      case "strong_sell":
        return {
          label: "STRONG SELL",
          color: "text-red-500",
          bg: "bg-red-500/10",
          border: "border-red-500",
          icon: TrendingDown,
          gradient: "from-red-500/20 to-transparent",
        }
      default:
        return {
          label: "HOLD",
          color: "text-yellow-500",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500",
          icon: Shield,
          gradient: "from-yellow-500/20 to-transparent",
        }
    }
  }

  const config = getRecommendationConfig()
  const Icon = config.icon
  const riskRewardRatio = (takeProfit - entry) / (entry - stopLoss)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <Card
        className={cn(
          "relative overflow-hidden border-2",
          config.border,
          "backdrop-blur-xl bg-card/50"
        )}
      >
        {/* Gradient Background */}
        <div className={cn("absolute inset-0 bg-gradient-to-br", config.gradient, "pointer-events-none")} />

        <div className="relative p-6 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <motion.div
              className="flex items-center gap-3"
              initial={{ x: -20 }}
              animate={{ x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className={cn("p-3 rounded-xl", config.bg)}>
                <Icon className={cn("w-6 h-6", config.color)} />
              </div>
              <div>
                <h3 className={cn("text-3xl font-bold tracking-tight", config.color)}>
                  {config.label}
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Confidence: {Math.round(confidence * 100)}%
                </p>
              </div>
            </motion.div>

            <Badge variant="outline" className="text-xs">
              Risk/Reward: {riskRewardRatio.toFixed(2)}
            </Badge>
          </div>

          {/* Metrics Grid */}
          <motion.div
            className="grid grid-cols-4 gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {/* Entry */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Target className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Entry</span>
              </div>
              <p className="text-lg font-bold">
                {numeral(entry).format("$0,0.00")}
              </p>
            </div>

            {/* Stop Loss */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Shield className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Stop Loss</span>
              </div>
              <p className="text-lg font-bold text-red-500">
                {numeral(stopLoss).format("$0,0.00")}
              </p>
            </div>

            {/* Take Profit */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <DollarSign className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Take Profit</span>
              </div>
              <p className="text-lg font-bold text-green-500">
                {numeral(takeProfit).format("$0,0.00")}
              </p>
            </div>

            {/* Position Size */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Percent className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Position Size</span>
              </div>
              <p className="text-lg font-bold">
                {positionSize}% of portfolio
              </p>
            </div>
          </motion.div>

          {/* Reasoning */}
          <motion.div
            className="pt-4 border-t border-border/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <h4 className="text-sm font-medium mb-2">Reasoning</h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {reasoning}
            </p>
          </motion.div>
        </div>

        {/* Animated border glow */}
        <motion.div
          className={cn(
            "absolute inset-0 rounded-lg opacity-50",
            "bg-gradient-to-r",
            config.gradient
          )}
          animate={{
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{ mixBlendMode: "overlay" }}
        />
      </Card>
    </motion.div>
  )
}
