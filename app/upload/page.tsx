"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, FileText, CheckCircle2, AlertCircle, Flag } from "lucide-react"
import { Label } from "@/components/ui/label"

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

export default function UploadPage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      // Check file type
      if (!selectedFile.name.endsWith(".csv") && !selectedFile.name.endsWith(".json")) {
        setError("Please upload a CSV or JSON file")
        setFile(null)
        return
      }
      setFile(selectedFile)
      setError("")
    }
  }

  const parseCSV = (text: string): Transaction[] => {
    const lines = text.trim().split("\n")
    const headers = lines[0].split(",").map((h) => h.trim())

    return lines.slice(1).map((line, index) => {
      const values = line.split(",").map((v) => v.trim())
      const transaction: any = {}

      headers.forEach((header, i) => {
        transaction[header] = values[i]
      })

      return {
        id: transaction.id || `TXN-${String(index + 1).padStart(3, "0")}`,
        timestamp: transaction.timestamp || new Date().toISOString(),
        fromBank: transaction.fromBank || transaction.from_bank || "",
        toBank: transaction.toBank || transaction.to_bank || "",
        amount: Number.parseFloat(transaction.amount) || 0,
        fromCurrency: transaction.fromCurrency || transaction.from_currency || "USD",
        toCurrency: transaction.toCurrency || transaction.to_currency || "USD",
        flagType: transaction.flagType || transaction.flag_type || "Suspicious Pattern",
      }
    })
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload")
      return
    }

    setIsUploading(true)
    setError("")

    try {
      const text = await file.text()
      let transactions: Transaction[]

      if (file.name.endsWith(".json")) {
        transactions = JSON.parse(text)
      } else {
        transactions = parseCSV(text)
      }

      // Validate transactions
      if (!Array.isArray(transactions) || transactions.length === 0) {
        throw new Error("Invalid file format or empty data")
      }

      // Store transactions in localStorage
      localStorage.setItem("transactions", JSON.stringify(transactions))

      setSuccess(true)

      // Redirect to dashboard after 1.5 seconds
      setTimeout(() => {
        router.push("/dashboard")
      }, 1500)
    } catch (err) {
      setError("Error processing file. Please check the format and try again.")
      console.error(err)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-4 dark:from-slate-950 dark:to-slate-900">
      <Card className="w-full max-w-2xl shadow-xl">
        <CardHeader className="space-y-3 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
            <Flag className="h-7 w-7 text-primary" />
          </div>
          <CardTitle className="text-2xl font-bold">Upload Transaction Data</CardTitle>
          <CardDescription className="text-base">
            Upload a CSV or JSON file containing flagged transaction records
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Area */}
          <div className="space-y-4">
            <Label htmlFor="file-upload" className="text-base font-medium">
              Transaction File
            </Label>
            <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/10 p-8 transition-colors hover:border-muted-foreground/50">
              <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
              <div className="text-center">
                <Label
                  htmlFor="file-upload"
                  className="cursor-pointer text-sm font-medium text-primary hover:underline"
                >
                  Click to upload
                </Label>
                <span className="text-sm text-muted-foreground"> or drag and drop</span>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">CSV or JSON files only</p>
              <input id="file-upload" type="file" accept=".csv,.json" onChange={handleFileChange} className="hidden" />
            </div>
          </div>

          {/* Selected File Display */}
          {file && (
            <div className="flex items-center gap-3 rounded-lg border bg-muted/50 p-4">
              <FileText className="h-8 w-8 text-primary" />
              <div className="flex-1">
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">{(file.size / 1024).toFixed(2)} KB</p>
              </div>
              {success && <CheckCircle2 className="h-6 w-6 text-green-600" />}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="flex items-center gap-2 rounded-lg bg-green-50 p-3 text-sm text-green-700 dark:bg-green-950/30 dark:text-green-400">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>File uploaded successfully! Redirecting to dashboard...</span>
            </div>
          )}

          {/* Upload Button */}
          <Button
            onClick={handleUpload}
            disabled={!file || isUploading || success}
            className="h-11 w-full text-base font-semibold"
          >
            {isUploading ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Processing...
              </>
            ) : success ? (
              <>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Upload Complete
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Upload & Continue
              </>
            )}
          </Button>

          {/* File Format Info */}
          <div className="rounded-lg bg-muted/50 p-4">
            <h4 className="mb-2 text-sm font-semibold">Expected File Format:</h4>
            <p className="mb-2 text-xs text-muted-foreground">
              CSV columns: id, timestamp, fromBank, toBank, amount, fromCurrency, toCurrency, flagType
            </p>
            <p className="text-xs text-muted-foreground">JSON: Array of transaction objects with the same fields</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}