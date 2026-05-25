import { sql } from '@/lib/db'
import { NextResponse } from 'next/server'

export async function POST() {
  try {
    // Create the muscle_readings table
    await sql`
      CREATE TABLE IF NOT EXISTS muscle_readings (
        id SERIAL PRIMARY KEY,
        signal_value DECIMAL(10, 2) NOT NULL,
        signal_percentage DECIMAL(5, 2) NOT NULL,
        status VARCHAR(20) NOT NULL,
        peak_value DECIMAL(10, 2),
        average_value DECIMAL(10, 2),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      )
    `

    // Create index for faster queries
    await sql`
      CREATE INDEX IF NOT EXISTS idx_muscle_readings_created_at 
      ON muscle_readings(created_at DESC)
    `

    return NextResponse.json({ success: true, message: 'Database setup complete' })
  } catch (error) {
    console.error('Database setup error:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to setup database' },
      { status: 500 }
    )
  }
}
