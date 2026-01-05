import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Analysis from "./pages/Analysis";
import Portfolio from "./pages/Portfolio";
import Backtest from "./pages/Backtest";
import Training from "./pages/Training";
import Settings from "./pages/Settings";
import Explainability from "./pages/Explainability";
import Alerts from "./pages/Alerts";
import RiskManagement from "./pages/RiskManagement";

function Router() {
  // make sure to consider if you need authentication for certain routes
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/dashboard"} component={Dashboard} />
      <Route path={"/analysis"} component={Analysis} />
      <Route path={"/portfolio"} component={Portfolio} />
      <Route path={"/backtest"} component={Backtest} />
      <Route path={"/training"} component={Training} />
      <Route path={"/settings"} component={Settings} />
      <Route path={"/explainability"} component={Explainability} />
      <Route path={"/alerts"} component={Alerts} />
      <Route path={"/risk"} component={RiskManagement} />
      <Route path={"/404"} component={NotFound} />
      {/* Final fallback route */}
      <Route component={NotFound} />
    </Switch>
  );
}

// NOTE: About Theme
// - First choose a default theme according to your design style (dark or light bg), than change color palette in index.css
//   to keep consistent foreground/background color across components
// - If you want to make theme switchable, pass `switchable` ThemeProvider and use `useTheme` hook

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider
        defaultTheme="dark"
        // switchable
      >
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
