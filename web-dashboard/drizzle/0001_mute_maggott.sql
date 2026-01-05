CREATE TABLE `analysis_history` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`symbol` varchar(10) NOT NULL,
	`analysisType` varchar(50) NOT NULL,
	`newsAgentResult` text,
	`technicalAgentResult` text,
	`fundamentalAgentResult` text,
	`strategistAgentResult` text,
	`supervisorResult` text,
	`recommendation` enum('buy','sell','hold'),
	`confidence` int,
	`targetPrice` varchar(20),
	`executionTime` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `analysis_history_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `portfolio_holdings` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`symbol` varchar(10) NOT NULL,
	`quantity` int NOT NULL,
	`averagePrice` varchar(20) NOT NULL,
	`currentPrice` varchar(20),
	`totalValue` varchar(20),
	`profitLoss` varchar(20),
	`profitLossPercent` varchar(10),
	`entryDate` timestamp NOT NULL,
	`exitDate` timestamp,
	`status` enum('active','closed') NOT NULL DEFAULT 'active',
	`notes` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `portfolio_holdings_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `training_logs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`sessionId` varchar(64) NOT NULL,
	`agentName` varchar(50) NOT NULL,
	`epoch` int NOT NULL,
	`loss` varchar(20),
	`accuracy` varchar(20),
	`precision` varchar(20),
	`recall` varchar(20),
	`f1Score` varchar(20),
	`learningRate` varchar(20),
	`batchSize` int,
	`modelVersion` varchar(50),
	`status` enum('running','completed','failed') NOT NULL,
	`duration` int,
	`notes` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `training_logs_id` PRIMARY KEY(`id`)
);
