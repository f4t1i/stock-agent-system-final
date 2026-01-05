import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  confidence: number; // 0-1
  size?: number; // diameter in pixels
  showLabel?: boolean;
  animated?: boolean;
}

export function ConfidenceGauge({
  confidence,
  size = 120,
  showLabel = true,
  animated = true,
}: ConfidenceGaugeProps) {
  const [displayValue, setDisplayValue] = useState(animated ? 0 : confidence);

  // Animate value on mount or change
  useEffect(() => {
    if (!animated) {
      setDisplayValue(confidence);
      return;
    }

    const duration = 1000; // 1 second
    const steps = 60;
    const increment = confidence / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      current = Math.min(confidence, step * increment);
      setDisplayValue(current);

      if (step >= steps) {
        clearInterval(timer);
        setDisplayValue(confidence);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [confidence, animated]);

  // Calculate gauge parameters
  const strokeWidth = size * 0.1;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (displayValue * circumference);

  // Get color based on confidence
  const getColor = () => {
    if (confidence >= 0.85) return "#10b981"; // emerald
    if (confidence >= 0.7) return "#22c55e"; // green
    if (confidence >= 0.5) return "#eab308"; // yellow
    if (confidence >= 0.3) return "#f59e0b"; // orange
    return "#ef4444"; // red
  };

  // Get label
  const getLabel = () => {
    if (confidence >= 0.85) return "Very High";
    if (confidence >= 0.7) return "High";
    if (confidence >= 0.5) return "Medium";
    if (confidence >= 0.3) return "Low";
    return "Very Low";
  };

  const color = getColor();

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          className="transform -rotate-90"
        >
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="none"
            className="text-muted/20"
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-300 ease-out"
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="font-bold"
            style={{
              fontSize: size * 0.25,
              color: color,
            }}
          >
            {Math.round(displayValue * 100)}%
          </span>
        </div>
      </div>

      {/* Label */}
      {showLabel && (
        <span
          className="text-sm font-medium"
          style={{ color: color }}
        >
          {getLabel()}
        </span>
      )}
    </div>
  );
}
