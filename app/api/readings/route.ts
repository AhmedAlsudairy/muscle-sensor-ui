import { sql } from '@/lib/db'
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '100')

    const readings = await sql`
      SELECT 
        id,
        signal_value,
        signal_percentage,
        status,
        peak_value,
        average_value,
        created_at
      FROM muscle_readings
      ORDER BY created_at DESC
      LIMIT ${limit}
    `

    // Get stats
    const stats = await sql`
      SELECT 
        COUNT(*) as total_readings,
        COALESCE(MAX(signal_percentage), 0) as max_signal,
        COALESCE(AVG(signal_percentage), 0) as avg_signal,
        MIN(created_at) as first_reading,
        MAX(created_at) as last_reading
      FROM muscle_readings
    `

    return NextResponse.json({
      success: true,
      readings: readings.reverse(), // Chronological order
      stats: stats[0]
    })
  } catch (error) {
    console.error('Failed to fetch readings:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch readings' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { signal_value, signal_percentage, status, peak_value, average_value } = body

    const result = await sql`
      INSERT INTO muscle_readings (signal_value, signal_percentage, status, peak_value, average_value)
      VALUES (${signal_value}, ${signal_percentage}, ${status}, ${peak_value}, ${average_value})
      RETURNING *
    `

    return NextResponse.json({ success: true, reading: result[0] })
  } catch (error) {
    console.error('Failed to save reading:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to save reading' },
      { status: 500 }
    )
  }
}

export async function DELETE() {
  try {
    await sql`DELETE FROM muscle_readings`
    return NextResponse.json({ success: true, message: 'All readings cleared' })
  } catch (error) {
    console.error('Failed to clear readings:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to clear readings' },
      { status: 500 }
    )
  }
}
