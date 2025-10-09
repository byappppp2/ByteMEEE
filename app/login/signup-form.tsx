"use client"

import type React from "react"

import { useState } from "react"
import supabase from "@/lib/supabaseClient"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, ShieldCheck } from "lucide-react"

interface SignupFormProps {
  onSignup: (user: any) => void
  onSwitchToLogin: () => void
}

export default function SignupForm({ onSignup, onSwitchToLogin }: SignupFormProps) {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [message, setMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setMessage("")
    setIsLoading(true)

    // Validate passwords match
    if (password !== confirmPassword) {
      setMessage("Passwords do not match")
      setIsLoading(false)
      return
    }

    // Validate password length
    if (password.length < 6) {
      setMessage("Password must be at least 6 characters long")
      setIsLoading(false)
      return
    }

    try {
      const { data, error } = await supabase.auth.signUp({
        email: email,
        password: password,
      })

      if (error) {
        setMessage(error.message)
        return
      }

      if (data?.user) {
        setMessage("Account created successfully! Please check your email to verify your account.")
        // Don't automatically log in - user needs to verify email first
        setTimeout(() => {
          onSwitchToLogin()
        }, 5000)
      }
    } catch (err) {
      setMessage("An unexpected error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-4 dark:from-slate-950 dark:to-slate-900">
      <Card className="w-full max-w-md shadow-xl">
        <CardHeader className="space-y-3 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
            <ShieldCheck className="h-7 w-7 text-primary" />
          </div>
          <CardTitle className="text-2xl font-bold">Transaction Monitor</CardTitle>
          <CardDescription className="text-base">Create your account</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="admin@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm Password</Label>
              <Input
                id="confirmPassword"
                type="password"
                placeholder="Enter your password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                disabled={isLoading}
                className="h-11"
              />
            </div>
            {message && (
              <div className={`flex items-center gap-2 rounded-lg p-3 text-sm ${
                message.includes("successfully") 
                  ? "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300" 
                  : "bg-destructive/10 text-destructive"
              }`}>
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{message}</span>
              </div>
            )}

            <Button type="submit" className="h-11 w-full text-base font-semibold" disabled={isLoading}>
              {isLoading ? (
                <>
                  <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                  Creating account...
                </>
              ) : (
                "Sign Up"
              )}
            </Button>
          </form>
          <div className="mt-4 text-center text-sm text-slate-600">
            <span>Already have an account? </span>
            <button 
              onClick={onSwitchToLogin}
              className="text-accent hover:text-accent/90 font-medium"
            >
              Sign in
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
