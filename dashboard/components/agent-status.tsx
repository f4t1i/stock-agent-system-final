"use client"

import { motion } from "framer-motion"
import { CheckCircle2, AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface AgentStatusProps {
  name: string
  signal: "buy" | "sell" | "hold"
  sentiment?: number // -2 to +2
  confidence: number // 0 to 1
  reasoning?: string
  delay?: number
}

export function AgentStatus({
  name,
  signal,
  sentiment,
  confidence,
  reasoning,
  delay = 0,
}: AgentStatusProps) {
  const getSignalConfig = () => {
    switch (signal) {
      case "buy":
        return {
          icon: TrendingUp,
          color: "text-green-500",
          bg: "bg-green-500/10",
          border: "border-green-500/20",
          badge: "success",
          label: sentiment ? `Bullish ${sentiment > 0 ? "+" : ""}${sentiment.toFixed(1)}` : "Buy Signal",
        }
      case "sell":
        return {
          icon: TrendingDown,
          color: "text-red-500",
          bg: "bg-red-500/10",
          border: "border-red-500/20",
          badge: "danger",
          label: sentiment ? `Bearish ${sentiment.toFixed(1)}` : "Sell Signal",
        }
      default:
        return {
          icon: Minus,
          color: "text-yellow-500",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/20",
          badge: "warning",
          label: "Hold",
        }
    }
  }

  const config = getSignalConfig()
  const Icon = config.icon
  const confidencePercent = Math.round(confidence * 100)

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={cn(
        "relative p-4 rounded-lg border",
        "backdrop-blur-sm bg-card/30",
        config.border,
        "hover:bg-card/50 transition-all duration-300",
        "group cursor-pointer"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={cn("p-1.5 rounded-md", config.bg)}>
            <Icon className={cn("w-4 h-4", config.color)} />
          </div>
          <span className="text-sm font-medium">{name}</span>
        </div>
        
        {signal !== "hold" && (
          <CheckCircle2 className={cn("w-4 h-4", config.color)} />
        )}
        {signal === "hold" && (
          <AlertCircle className="w-4 h-4 text-yellow-500" />
        )}
      </div>

      {/* Signal */}
      <div className="space-y-2">
        <Badge
          variant={config.badge as any}
          className="font-medium"
        >
          {config.label}
        </Badge>

        {/* Confidence Bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-medium">{confidencePercent}%</span>
          </div>
          <Progress
            value={confidencePercent}
            className="h-1.5"
            indicatorClassName={cn(
              signal === "buy" && "bg-green-500",
              signal === "sell" && "bg-red-500",
              signal === "hold" && "bg-yellow-500"
            )}
          />
        </div>
      </div>

      {/* Reasoning (on hover) */}
      {reasoning && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          whileHover={{ opacity: 1, height: "auto" }}
          className="mt-3 pt-3 border-t border-border/50 text-xs text-muted-foreground overflow-hidden"
        >
          {reasoning}
        </motion.div>
      )}

      {/* Glow effect */}
      <div
        className={cn(
          "absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none",
          "bg-gradient-to-br",
          signal === "buy" && "from-green-500/10 to-transparent",
          signal === "sell" && "from-red-500/10 to-transparent",
          signal === "hold" && "from-yellow-500/10 to-transparent"
        )}
      />
    </motion.div>
  )
}
