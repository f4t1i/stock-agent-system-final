"use client"

import { motion } from "framer-motion"
import { LucideIcon } from "lucide-react"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import numeral from "numeral"

interface MetricCardProps {
  title: string
  value: number | string
  change?: number
  icon?: LucideIcon
  format?: "currency" | "percent" | "number"
  trend?: "up" | "down" | "neutral"
  delay?: number
  className?: string
}

export function MetricCard({
  title,
  value,
  change,
  icon: Icon,
  format = "number",
  trend = "neutral",
  delay = 0,
  className,
}: MetricCardProps) {
  const formatValue = (val: number | string) => {
    if (typeof val === "string") return val
    
    switch (format) {
      case "currency":
        return numeral(val).format("$0,0.00")
      case "percent":
        return numeral(val / 100).format("0.00%")
      default:
        return numeral(val).format("0,0")
    }
  }

  const getTrendColor = () => {
    switch (trend) {
      case "up":
        return "text-green-500"
      case "down":
        return "text-red-500"
      default:
        return "text-gray-500"
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
    >
      <Card
        className={cn(
          "relative overflow-hidden",
          "backdrop-blur-xl bg-card/50 border-border/50",
          "hover:bg-card/70 transition-all duration-300",
          "hover:scale-[1.02] hover:shadow-lg",
          className
        )}
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-medium text-muted-foreground">
              {title}
            </p>
            {Icon && (
              <div className="p-2 rounded-lg bg-primary/10">
                <Icon className="w-4 h-4 text-primary" />
              </div>
            )}
          </div>

          {/* Value */}
          <div className="space-y-2">
            <motion.p
              className="text-3xl font-bold tracking-tight"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, delay: delay + 0.2 }}
            >
              {formatValue(value)}
            </motion.p>

            {/* Change */}
            {change !== undefined && (
              <motion.div
                className="flex items-center gap-1"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: delay + 0.3 }}
              >
                <span className={cn("text-sm font-medium", getTrendColor())}>
                  {change > 0 ? "+" : ""}
                  {numeral(change / 100).format("0.00%")}
                </span>
                <span className="text-xs text-muted-foreground">
                  vs. yesterday
                </span>
              </motion.div>
            )}
          </div>

          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent pointer-events-none" />
        </div>
      </Card>
    </motion.div>
  )
}
