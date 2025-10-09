"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { TransactionsDashboard } from "@/components/transactions-dashboard"
import supabase from "@/lib/supabaseClient"
import LoginForm from "@/app/login/login-form"
import SignupForm from "@/app/login/signup-form"
import UploadPage from "@/app/upload/page"

export default function HomePage() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [authView, setAuthView] = useState<"login" | "signup">("login")
const [uploadCompleted, setUploadCompleted] = useState(false)

  useEffect(() => {
    // Check if user is authenticated
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (session?.user) {
        setUser(session.user)
        setIsAuthenticated(true)
        const uploaded = localStorage.getItem("transactions")
        if (uploaded){
          setUploadCompleted(true)
        }
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const handleLogin = (userData: any) => {
    setUser(userData)
    setIsAuthenticated(true)
  }

  const handleSignup = (userData: any) => {
    setUser(userData)
    setIsAuthenticated(true)
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    setUser(null)
    setIsAuthenticated(false)
    setUploadCompleted(false)
    localStorage.removeItem("transactions")
  }

    useEffect(() => {
    const init = async () => {
      const { data } = await supabase.auth.getSession()
      if (data.session?.user) {
        setUser(data.session.user)
      }
    }
    init()

    const { data: subscription } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user ?? null)
      setIsAuthenticated(!!session?.user)
      
      // Handle OAuth callback
      if (event === 'SIGNED_IN' && session?.user) {
        setUser(session.user)
        setIsAuthenticated(true)
      } else if (event === 'SIGNED_OUT') {
        setUser(null)
        setIsAuthenticated(false)
        setUploadCompleted(false)
      }
    })

    return () => {
      subscription.subscription?.unsubscribe()
    }
  }, [])

  if (!user) {
    if (authView === "login") {
      return <LoginForm onLogin={handleLogin} onSwitchToSignup={() => setAuthView("signup")} />
    } else {
      return <SignupForm onSignup={handleSignup} onSwitchToLogin={() => setAuthView("login")} />
    }
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  if (isAuthenticated && !uploadCompleted) {
    return <UploadPage />
  }

  // 3️⃣ If logged in and upload done → dashboard
  if (isAuthenticated && uploadCompleted) {
    console.log("hello")
    return <TransactionsDashboard onLogout={handleLogout} />
  }
 return null

  // return <TransactionsDashboard onLogout={handleLogout} />
}
