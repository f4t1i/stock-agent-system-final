/**
 * Dashboard Grid Layout Components
 *
 * Flexible grid system for organizing dashboard widgets and charts.
 * Supports responsive layouts, drag-and-drop (future), and customization.
 */

import React, { ReactNode } from 'react';

export interface DashboardGridProps {
  columns?: 1 | 2 | 3 | 4 | 6 | 12;
  gap?: number;
  className?: string;
  children: ReactNode;
}

export function DashboardGrid({
  columns = 12,
  gap = 4,
  className = '',
  children,
}: DashboardGridProps) {
  const gridColsClass = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 lg:grid-cols-2',
    3: 'grid-cols-1 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
    6: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6',
    12: 'grid-cols-12',
  }[columns];

  const gapClass = `gap-${gap}`;

  return (
    <div className={`grid ${gridColsClass} ${gapClass} ${className}`}>
      {children}
    </div>
  );
}

export interface GridItemProps {
  colSpan?: number;
  rowSpan?: number;
  className?: string;
  children: ReactNode;
}

export function GridItem({
  colSpan = 1,
  rowSpan = 1,
  className = '',
  children,
}: GridItemProps) {
  const colSpanClass = `col-span-${colSpan}`;
  const rowSpanClass = rowSpan > 1 ? `row-span-${rowSpan}` : '';

  return (
    <div className={`${colSpanClass} ${rowSpanClass} ${className}`}>
      {children}
    </div>
  );
}

export interface WidgetProps {
  title?: string;
  subtitle?: string;
  headerActions?: ReactNode;
  footer?: ReactNode;
  padding?: boolean;
  className?: string;
  children: ReactNode;
}

export function Widget({
  title,
  subtitle,
  headerActions,
  footer,
  padding = true,
  className = '',
  children,
}: WidgetProps) {
  return (
    <div className={`bg-gray-800 rounded-lg overflow-hidden ${className}`}>
      {(title || subtitle || headerActions) && (
        <div className="border-b border-gray-700 px-4 py-3">
          <div className="flex items-start justify-between">
            <div>
              {title && (
                <h3 className="text-white text-lg font-semibold">{title}</h3>
              )}
              {subtitle && (
                <p className="text-gray-400 text-sm mt-1">{subtitle}</p>
              )}
            </div>
            {headerActions && <div className="flex items-center gap-2">{headerActions}</div>}
          </div>
        </div>
      )}

      <div className={padding ? 'p-4' : ''}>{children}</div>

      {footer && (
        <div className="border-t border-gray-700 px-4 py-3 bg-gray-750">
          {footer}
        </div>
      )}
    </div>
  );
}

export interface MetricGridProps {
  metrics: Array<{
    label: string;
    value: string | number;
    change?: number;
    prefix?: string;
    suffix?: string;
    trend?: 'up' | 'down' | 'neutral';
    icon?: ReactNode;
  }>;
  columns?: 2 | 3 | 4;
  className?: string;
}

export function MetricGrid({
  metrics,
  columns = 4,
  className = '',
}: MetricGridProps) {
  const gridColsClass = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
  }[columns];

  return (
    <div className={`grid ${gridColsClass} gap-4 ${className}`}>
      {metrics.map((metric, index) => {
        const changeColor =
          metric.trend === 'up'
            ? 'text-green-400'
            : metric.trend === 'down'
            ? 'text-red-400'
            : 'text-gray-400';

        return (
          <div key={index} className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-gray-400 text-sm">{metric.label}</p>
              {metric.icon && <div className="text-gray-500">{metric.icon}</div>}
            </div>

            <div className="flex items-baseline gap-2">
              <p className="text-white text-2xl font-bold">
                {metric.prefix}
                {typeof metric.value === 'number'
                  ? metric.value.toLocaleString()
                  : metric.value}
                {metric.suffix}
              </p>
            </div>

            {metric.change !== undefined && (
              <p className={`text-sm mt-1 ${changeColor}`}>
                {metric.change >= 0 ? '+' : ''}
                {metric.change.toFixed(2)}%
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export interface TabsProps {
  tabs: Array<{
    id: string;
    label: string;
    icon?: ReactNode;
    badge?: number;
  }>;
  activeTab: string;
  onChange: (tabId: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onChange, className = '' }: TabsProps) {
  return (
    <div className={`border-b border-gray-700 ${className}`}>
      <nav className="flex -mb-px">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab;

          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-3 border-b-2 font-medium text-sm
                transition-colors
                ${
                  isActive
                    ? 'border-blue-500 text-blue-500'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                }
              `}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span
                  className={`
                  px-2 py-0.5 rounded-full text-xs font-semibold
                  ${
                    isActive
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-700 text-gray-300'
                  }
                `}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );
}

export interface SplitPanelProps {
  left: ReactNode;
  right: ReactNode;
  leftWidth?: string;
  rightWidth?: string;
  gap?: number;
  className?: string;
}

export function SplitPanel({
  left,
  right,
  leftWidth = '300px',
  rightWidth = 'auto',
  gap = 4,
  className = '',
}: SplitPanelProps) {
  return (
    <div className={`flex gap-${gap} ${className}`}>
      <div style={{ width: leftWidth, flexShrink: 0 }}>{left}</div>
      <div style={{ width: rightWidth, flex: 1 }}>{right}</div>
    </div>
  );
}

export interface AccordionItemProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  icon?: ReactNode;
}

export function AccordionItem({
  title,
  children,
  defaultOpen = false,
  icon,
}: AccordionItemProps) {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);

  return (
    <div className="border-b border-gray-700 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-750 transition-colors"
      >
        <div className="flex items-center gap-2">
          {icon && <div className="text-gray-400">{icon}</div>}
          <span className="text-white font-medium">{title}</span>
        </div>
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform ${
            isOpen ? 'transform rotate-180' : ''
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="px-4 py-3 bg-gray-750">
          {children}
        </div>
      )}
    </div>
  );
}

export interface AccordionProps {
  items: Array<{
    title: string;
    content: ReactNode;
    defaultOpen?: boolean;
    icon?: ReactNode;
  }>;
  className?: string;
}

export function Accordion({ items, className = '' }: AccordionProps) {
  return (
    <div className={`bg-gray-800 rounded-lg overflow-hidden ${className}`}>
      {items.map((item, index) => (
        <AccordionItem
          key={index}
          title={item.title}
          defaultOpen={item.defaultOpen}
          icon={item.icon}
        >
          {item.content}
        </AccordionItem>
      ))}
    </div>
  );
}

export interface StatusBadgeProps {
  status: 'success' | 'warning' | 'error' | 'info' | 'neutral';
  text: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function StatusBadge({
  status,
  text,
  size = 'md',
  className = '',
}: StatusBadgeProps) {
  const colors = {
    success: 'bg-green-900 text-green-300 border-green-700',
    warning: 'bg-yellow-900 text-yellow-300 border-yellow-700',
    error: 'bg-red-900 text-red-300 border-red-700',
    info: 'bg-blue-900 text-blue-300 border-blue-700',
    neutral: 'bg-gray-700 text-gray-300 border-gray-600',
  }[status];

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  }[size];

  return (
    <span
      className={`inline-flex items-center rounded-full border font-semibold ${colors} ${sizes} ${className}`}
    >
      {text}
    </span>
  );
}

export interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function ProgressBar({
  value,
  max = 100,
  label,
  showPercentage = true,
  color = 'blue',
  size = 'md',
  className = '',
}: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100);

  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
  }[color];

  const heights = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  }[size];

  return (
    <div className={className}>
      {(label || showPercentage) && (
        <div className="flex justify-between text-sm mb-1">
          {label && <span className="text-gray-300">{label}</span>}
          {showPercentage && (
            <span className="text-gray-400 font-mono">{percentage.toFixed(0)}%</span>
          )}
        </div>
      )}
      <div className={`w-full bg-gray-700 rounded-full overflow-hidden ${heights}`}>
        <div
          className={`${colors} ${heights} rounded-full transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
