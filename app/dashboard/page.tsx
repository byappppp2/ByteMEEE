"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { TransactionsDashboard } from "@/components/transactions-dashboard"

export default function DashboardPage() {
  const router = useRouter()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if user is authenticated
    const authStatus = localStorage.getItem("isAuthenticated")
    const uploaded = localStorage.getItem("transactions")
  if (authStatus !== "true") {
    router.push("/login")
  } else if (!uploaded) {
    router.push("/upload")
  } else {
    setIsAuthenticated(true)
  }

  setIsLoading(false)
  }, [router])

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated")
    localStorage.removeItem("transactions")
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  return <TransactionsDashboard onLogout={handleLogout} />
}