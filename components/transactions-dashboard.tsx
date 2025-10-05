"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { LogOut, Search, AlertTriangle, TrendingUp, DollarSign, Flag, Loader2, FileText, Sparkles } from "lucide-react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"

interface Transaction {
  id: string
  timestamp: string
  fromBank: string
  toBank: string
  amount: number
  fromCurrency: string
  toCurrency: string
  flagType: "High Risk" | "Suspicious Pattern" | "Unusual Amount" | "Velocity Check"
}

const mockTransactions: Transaction[] = [
  {
    id: "TXN-001",
    timestamp: "2025-01-05 14:32:15",
    fromBank: "Chase Bank",
    toBank: "Wells Fargo",
    amount: 125000,
    fromCurrency: "USD",
    toCurrency: "USD",
    flagType: "High Risk",
  },
  {
    id: "TXN-002",
    timestamp: "2025-01-05 13:18:42",
    fromBank: "Bank of America",
    toBank: "HSBC International",
    amount: 45000,
    fromCurrency: "USD",
    toCurrency: "EUR",
    flagType: "Suspicious Pattern",
  },
  {
    id: "TXN-003",
    timestamp: "2025-01-05 12:05:33",
    fromBank: "Citibank",
    toBank: "Deutsche Bank",
    amount: 89500,
    fromCurrency: "USD",
    toCurrency: "GBP",
    flagType: "Unusual Amount",
  },
  {
    id: "TXN-004",
    timestamp: "2025-01-05 11:47:21",
    fromBank: "TD Bank",
    toBank: "Barclays",
    amount: 32000,
    fromCurrency: "CAD",
    toCurrency: "USD",
    flagType: "Velocity Check",
  },
  {
    id: "TXN-005",
    timestamp: "2025-01-05 10:22:08",
    fromBank: "PNC Bank",
    toBank: "Standard Chartered",
    amount: 156000,
    fromCurrency: "USD",
    toCurrency: "SGD",
    flagType: "High Risk",
  },
  {
    id: "TXN-006",
    timestamp: "2025-01-05 09:15:54",
    fromBank: "Capital One",
    toBank: "BNP Paribas",
    amount: 67800,
    fromCurrency: "USD",
    toCurrency: "EUR",
    flagType: "Suspicious Pattern",
  },
  {
    id: "TXN-007",
    timestamp: "2025-01-05 08:43:12",
    fromBank: "US Bank",
    toBank: "Santander",
    amount: 94200,
    fromCurrency: "USD",
    toCurrency: "EUR",
    flagType: "Unusual Amount",
  },
  {
    id: "TXN-008",
    timestamp: "2025-01-04 16:28:37",
    fromBank: "BBVA",
    toBank: "JPMorgan Chase",
    amount: 41500,
    fromCurrency: "EUR",
    toCurrency: "USD",
    flagType: "Velocity Check",
  },
]

const flagTypeColors = {
  "High Risk": "destructive",
  "Suspicious Pattern": "default",
  "Unusual Amount": "secondary",
  "Velocity Check": "outline",
} as const

export function TransactionsDashboard({ onLogout }: { onLogout: () => void }) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [complianceReport, setComplianceReport] = useState<string | null>(null)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)

  const filteredTransactions = mockTransactions.filter(
    (transaction) =>
      transaction.fromBank.toLowerCase().includes(searchQuery.toLowerCase()) ||
      transaction.toBank.toLowerCase().includes(searchQuery.toLowerCase()) ||
      transaction.flagType.toLowerCase().includes(searchQuery.toLowerCase()) ||
      transaction.id.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const totalFlagged = mockTransactions.length
  const highRiskCount = mockTransactions.filter((t) => t.flagType === "High Risk").length
  const totalAmount = mockTransactions.reduce((sum, t) => sum + t.amount, 0)

  const handleTransactionClick = (transaction: Transaction) => {
    setSelectedTransaction(transaction)
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
    }, 2000)
  }

  const generateComplianceReport = async () => {
    setIsGeneratingReport(true)
    setComplianceReport(null)
    try {
      const response = await fetch("/api/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transactions: mockTransactions }),
      })
      const data = await response.json()
      setComplianceReport(data.report)
    } catch (error) {
      console.error("Error generating compliance report:", error)
      setComplianceReport("Error generating report. Please try again.")
    } finally {
      setIsGeneratingReport(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Header */}
      <header className="border-b bg-background">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
              <Flag className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-none">Transaction Monitor</h1>
              <p className="text-xs text-muted-foreground">Flagged Transactions Dashboard</p>
            </div>
          </div>
          <Button variant="outline" onClick={onLogout} className="gap-2 bg-transparent">
            <LogOut className="h-4 w-4" />
            Sign Out
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-4 md:p-6 lg:p-8">
        {/* Stats Cards */}
        <div className="mb-6 grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Flagged</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{totalFlagged}</div>
              <p className="text-xs text-muted-foreground">transactions require review</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">High Risk</CardTitle>
              <TrendingUp className="h-4 w-4 text-destructive" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-destructive">{highRiskCount}</div>
              <p className="text-xs text-muted-foreground">critical alerts</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Amount</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">${(totalAmount / 1000).toFixed(0)}K</div>
              <p className="text-xs text-muted-foreground">across all flagged transactions</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-primary/5 to-primary/10">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">AI Report</CardTitle>
              <Sparkles className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <Button
                onClick={generateComplianceReport}
                disabled={isGeneratingReport}
                className="w-full gap-2"
                size="sm"
              >
                {isGeneratingReport ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <FileText className="h-4 w-4" />
                    Generate Report
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Transactions Table */}
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <CardTitle>Flagged Transactions</CardTitle>
                <CardDescription>Monitor and review transactions that require attention</CardDescription>
              </div>
              <div className="relative w-full md:w-80">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search transactions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[140px]">Timestamp</TableHead>
                    <TableHead>From Bank</TableHead>
                    <TableHead>To Bank</TableHead>
                    <TableHead className="text-right">Amount</TableHead>
                    <TableHead>Currencies</TableHead>
                    <TableHead>Flag Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredTransactions.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="h-32 text-center text-muted-foreground">
                        No transactions found
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredTransactions.map((transaction) => (
                      <TableRow
                        key={transaction.id}
                        onClick={() => handleTransactionClick(transaction)}
                        className="cursor-pointer"
                      >
                        <TableCell className="font-mono text-xs">{transaction.timestamp}</TableCell>
                        <TableCell className="font-medium">{transaction.fromBank}</TableCell>
                        <TableCell className="font-medium">{transaction.toBank}</TableCell>
                        <TableCell className="text-right font-semibold">
                          ${transaction.amount.toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <span className="font-mono text-sm">
                            {transaction.fromCurrency} → {transaction.toCurrency}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Badge variant={flagTypeColors[transaction.flagType]}>{transaction.flagType}</Badge>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Transaction Details Dialog */}
        <Dialog open={!!selectedTransaction} onOpenChange={() => setSelectedTransaction(null)}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Transaction Details</DialogTitle>
              <DialogDescription>Review the details of the selected transaction.</DialogDescription>
            </DialogHeader>
            {isLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
              </div>
            ) : (
              selectedTransaction && (
                <div className="grid gap-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Transaction ID</p>
                      <p className="text-lg font-semibold">{selectedTransaction.id}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Timestamp</p>
                      <p className="text-lg font-semibold">{selectedTransaction.timestamp}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">From Bank</p>
                      <p className="text-lg font-semibold">{selectedTransaction.fromBank}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">To Bank</p>
                      <p className="text-lg font-semibold">{selectedTransaction.toBank}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Amount</p>
                      <p className="text-lg font-semibold">${selectedTransaction.amount.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Currencies</p>
                      <p className="text-lg font-semibold">
                        {selectedTransaction.fromCurrency} → {selectedTransaction.toCurrency}
                      </p>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Flag Type</p>
                    <Badge variant={flagTypeColors[selectedTransaction.flagType]} className="mt-1">
                      {selectedTransaction.flagType}
                    </Badge>
                  </div>
                </div>
              )
            )}
          </DialogContent>
        </Dialog>

        <Dialog open={!!complianceReport} onOpenChange={() => setComplianceReport(null)}>
          <DialogContent className="max-w-3xl max-h-[80vh]">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                AI-Generated Compliance Report
              </DialogTitle>
              <DialogDescription>Automated analysis of flagged transactions for compliance review</DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh] pr-4">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{complianceReport}</div>
              </div>
            </ScrollArea>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  )
}
