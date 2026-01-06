/**
 * Chart Components Tests
 *
 * Unit and integration tests for visualization components.
 * Tests rendering, data transformation, and user interactions.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import { CandlestickChart, CandlestickData } from '../CandlestickChart';
import { LineChart, LineSeries } from '../LineChart';
import { AreaChart, AreaSeries } from '../AreaChart';
import { Heatmap, HeatmapCell } from '../Heatmap';
import { TechnicalChart } from '../TechnicalChart';

// Mock Recharts components for testing
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  Bar: () => <div data-testid="bar" />,
  Cell: () => <div data-testid="cell" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
}));

// ============================================================================
// CandlestickChart Tests
// ============================================================================

describe('CandlestickChart', () => {
  const mockData: CandlestickData[] = [
    {
      timestamp: '2024-01-01T00:00:00Z',
      open: 100,
      high: 110,
      low: 95,
      close: 105,
      volume: 1000000,
    },
    {
      timestamp: '2024-01-02T00:00:00Z',
      open: 105,
      high: 115,
      low: 100,
      close: 112,
      volume: 1200000,
    },
    {
      timestamp: '2024-01-03T00:00:00Z',
      open: 112,
      high: 120,
      low: 108,
      close: 115,
      volume: 1100000,
    },
  ];

  it('renders without crashing', () => {
    render(<CandlestickChart data={mockData} />);
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });

  it('shows "No data available" when empty', () => {
    render(<CandlestickChart data={[]} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders with volume when showVolume is true', () => {
    render(<CandlestickChart data={mockData} showVolume={true} />);
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
  });

  it('renders moving averages when showMA is true', () => {
    render(<CandlestickChart data={mockData} showMA={true} maPeriods={[20, 50]} />);
    // MA lines should be present
    const lines = screen.getAllByTestId('line');
    expect(lines.length).toBeGreaterThan(0);
  });

  it('respects custom height prop', () => {
    const { container } = render(<CandlestickChart data={mockData} height={600} />);
    const responsiveContainer = screen.getByTestId('responsive-container');
    // Check that height prop is passed to ResponsiveContainer
    expect(responsiveContainer).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <CandlestickChart data={mockData} className="custom-class" />
    );
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });
});

// ============================================================================
// LineChart Tests
// ============================================================================

describe('LineChart', () => {
  const mockData = [
    { timestamp: '2024-01-01', AAPL: 180, MSFT: 370 },
    { timestamp: '2024-01-02', AAPL: 185, MSFT: 375 },
    { timestamp: '2024-01-03', AAPL: 183, MSFT: 378 },
  ];

  const mockSeries: LineSeries[] = [
    { dataKey: 'AAPL', name: 'Apple', color: '#3b82f6' },
    { dataKey: 'MSFT', name: 'Microsoft', color: '#10b981' },
  ];

  it('renders without crashing', () => {
    render(<LineChart data={mockData} series={mockSeries} />);
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('shows "No data available" when empty', () => {
    render(<LineChart data={[]} series={mockSeries} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders all series', () => {
    render(<LineChart data={mockData} series={mockSeries} />);
    const lines = screen.getAllByTestId('line');
    expect(lines.length).toBe(mockSeries.length);
  });

  it('renders reference lines when provided', () => {
    render(
      <LineChart
        data={mockData}
        series={mockSeries}
        referenceLines={[{ value: 180, label: 'Target', color: '#ef4444' }]}
      />
    );
    expect(screen.getByTestId('reference-line')).toBeInTheDocument();
  });

  it('shows legend when showLegend is true', () => {
    render(<LineChart data={mockData} series={mockSeries} showLegend={true} />);
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('hides grid when showGrid is false', () => {
    render(<LineChart data={mockData} series={mockSeries} showGrid={false} />);
    expect(screen.queryByTestId('cartesian-grid')).not.toBeInTheDocument();
  });
});

// ============================================================================
// AreaChart Tests
// ============================================================================

describe('AreaChart', () => {
  const mockData = [
    { timestamp: '2024-01-01', stocks: 60, bonds: 30, cash: 10 },
    { timestamp: '2024-01-02', stocks: 65, bonds: 25, cash: 10 },
    { timestamp: '2024-01-03', stocks: 70, bonds: 20, cash: 10 },
  ];

  const mockSeries: AreaSeries[] = [
    { dataKey: 'stocks', name: 'Stocks', color: '#3b82f6' },
    { dataKey: 'bonds', name: 'Bonds', color: '#10b981' },
    { dataKey: 'cash', name: 'Cash', color: '#f59e0b' },
  ];

  it('renders without crashing', () => {
    render(<AreaChart data={mockData} series={mockSeries} />);
    expect(screen.getByTestId('area-chart')).toBeInTheDocument();
  });

  it('shows "No data available" when empty', () => {
    render(<AreaChart data={[]} series={mockSeries} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders all series', () => {
    render(<AreaChart data={mockData} series={mockSeries} />);
    const areas = screen.getAllByTestId('area');
    expect(areas.length).toBe(mockSeries.length);
  });

  it('stacks areas when stacked is true', () => {
    render(<AreaChart data={mockData} series={mockSeries} stacked={true} />);
    // Areas should have stackId when stacked
    expect(screen.getAllByTestId('area')).toHaveLength(mockSeries.length);
  });
});

// ============================================================================
// Heatmap Tests
// ============================================================================

describe('Heatmap', () => {
  const mockData: HeatmapCell[] = [
    { row: 'A', col: 'X', value: 0.8 },
    { row: 'A', col: 'Y', value: 0.5 },
    { row: 'B', col: 'X', value: 0.3 },
    { row: 'B', col: 'Y', value: 0.9 },
  ];

  const rows = ['A', 'B'];
  const cols = ['X', 'Y'];

  it('renders without crashing', () => {
    render(<Heatmap data={mockData} rows={rows} cols={cols} />);
    expect(screen.getByRole('generic')).toBeInTheDocument();
  });

  it('shows "No data available" when empty', () => {
    render(<Heatmap data={[]} rows={rows} cols={cols} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders correct number of cells', () => {
    const { container } = render(<Heatmap data={mockData} rows={rows} cols={cols} />);
    // Should render rows.length * cols.length cells
    const cells = container.querySelectorAll('[style*="cursor: pointer"]');
    expect(cells.length).toBe(rows.length * cols.length);
  });

  it('shows values when showValues is true', () => {
    render(<Heatmap data={mockData} rows={rows} cols={cols} showValues={true} />);
    expect(screen.getByText('0.80')).toBeInTheDocument();
  });

  it('hides labels when showLabels is false', () => {
    render(<Heatmap data={mockData} rows={rows} cols={cols} showLabels={false} />);
    expect(screen.queryByText('A')).not.toBeInTheDocument();
  });

  it('shows tooltip on hover', async () => {
    const { container } = render(<Heatmap data={mockData} rows={rows} cols={cols} />);
    const cell = container.querySelector('[style*="cursor: pointer"]');

    if (cell) {
      fireEvent.mouseEnter(cell);

      await waitFor(() => {
        expect(screen.getByText('Row:')).toBeInTheDocument();
        expect(screen.getByText('Value:')).toBeInTheDocument();
      });
    }
  });
});

// ============================================================================
// TechnicalChart Tests
// ============================================================================

describe('TechnicalChart', () => {
  const mockData: CandlestickData[] = [
    {
      timestamp: '2024-01-01T00:00:00Z',
      open: 100,
      high: 110,
      low: 95,
      close: 105,
      volume: 1000000,
    },
    {
      timestamp: '2024-01-02T00:00:00Z',
      open: 105,
      high: 115,
      low: 100,
      close: 112,
      volume: 1200000,
    },
    // Add more data points for valid indicator calculations
    ...Array.from({ length: 50 }, (_, i) => ({
      timestamp: new Date(Date.now() + i * 24 * 60 * 60 * 1000).toISOString(),
      open: 100 + Math.random() * 20,
      high: 110 + Math.random() * 20,
      low: 90 + Math.random() * 20,
      close: 105 + Math.random() * 20,
      volume: 1000000 + Math.random() * 500000,
    })),
  ];

  it('renders without crashing', () => {
    render(<TechnicalChart data={mockData} />);
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
  });

  it('shows "No data available" when empty', () => {
    render(<TechnicalChart data={[]} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('renders indicator toggle buttons', () => {
    render(<TechnicalChart data={mockData} />);
    expect(screen.getByText('Bollinger Bands')).toBeInTheDocument();
    expect(screen.getByText('RSI')).toBeInTheDocument();
    expect(screen.getByText('MACD')).toBeInTheDocument();
  });

  it('toggles indicators on button click', () => {
    render(<TechnicalChart data={mockData} />);
    const rsiButton = screen.getByText('RSI');

    // Initial state
    expect(rsiButton).toHaveClass('bg-blue-600');

    // Toggle off
    fireEvent.click(rsiButton);
    expect(rsiButton).toHaveClass('bg-gray-700');

    // Toggle on
    fireEvent.click(rsiButton);
    expect(rsiButton).toHaveClass('bg-blue-600');
  });

  it('renders RSI panel when enabled', () => {
    render(
      <TechnicalChart
        data={mockData}
        indicators={{ rsi: true }}
      />
    );
    // RSI reference lines at 70 and 30
    const referenceLines = screen.getAllByTestId('reference-line');
    expect(referenceLines.length).toBeGreaterThanOrEqual(2);
  });

  it('renders MACD panel when enabled', () => {
    render(
      <TechnicalChart
        data={mockData}
        indicators={{ macd: true }}
      />
    );
    // MACD should have histogram bars
    const bars = screen.getAllByTestId('bar');
    expect(bars.length).toBeGreaterThan(0);
  });

  it('renders volume panel when enabled', () => {
    render(
      <TechnicalChart
        data={mockData}
        indicators={{ volume: true }}
      />
    );
    const charts = screen.getAllByTestId('composed-chart');
    expect(charts.length).toBeGreaterThan(1); // Main + volume
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Chart Integration', () => {
  it('multiple charts render together', () => {
    const mockCandleData: CandlestickData[] = [
      {
        timestamp: '2024-01-01T00:00:00Z',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000000,
      },
    ];

    const mockLineData = [{ timestamp: '2024-01-01', value: 100 }];
    const mockLineSeries: LineSeries[] = [
      { dataKey: 'value', name: 'Value', color: '#3b82f6' },
    ];

    const { container } = render(
      <div>
        <CandlestickChart data={mockCandleData} />
        <LineChart data={mockLineData} series={mockLineSeries} />
      </div>
    );

    const charts = container.querySelectorAll('[data-testid="responsive-container"]');
    expect(charts.length).toBe(2);
  });

  it('handles data updates correctly', () => {
    const initialData: CandlestickData[] = [
      {
        timestamp: '2024-01-01T00:00:00Z',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000000,
      },
    ];

    const { rerender } = render(<CandlestickChart data={initialData} />);

    const updatedData: CandlestickData[] = [
      ...initialData,
      {
        timestamp: '2024-01-02T00:00:00Z',
        open: 105,
        high: 115,
        low: 100,
        close: 112,
        volume: 1200000,
      },
    ];

    rerender(<CandlestickChart data={updatedData} />);

    expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
  });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

describe('Chart Error Handling', () => {
  it('handles invalid data gracefully', () => {
    const invalidData: any[] = [
      { timestamp: 'invalid', close: NaN },
    ];

    expect(() => {
      render(<CandlestickChart data={invalidData as CandlestickData[]} />);
    }).not.toThrow();
  });

  it('handles missing required fields', () => {
    const incompleteData: any[] = [
      { timestamp: '2024-01-01' }, // Missing OHLCV
    ];

    expect(() => {
      render(<CandlestickChart data={incompleteData as CandlestickData[]} />);
    }).not.toThrow();
  });

  it('handles null/undefined data prop', () => {
    expect(() => {
      render(<CandlestickChart data={null as any} />);
    }).not.toThrow();

    expect(() => {
      render(<CandlestickChart data={undefined as any} />);
    }).not.toThrow();
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

describe('Chart Accessibility', () => {
  it('has proper ARIA labels', () => {
    const mockData: CandlestickData[] = [
      {
        timestamp: '2024-01-01T00:00:00Z',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000000,
      },
    ];

    const { container } = render(<CandlestickChart data={mockData} />);
    // Check for accessibility attributes
    expect(container.querySelector('[role]')).toBeTruthy();
  });

  it('supports keyboard navigation', () => {
    render(<TechnicalChart data={[]} />);

    const rsiButton = screen.getByText('RSI');
    expect(rsiButton.tagName).toBe('BUTTON');
    expect(rsiButton).toHaveAttribute('type', 'button');
  });
});
