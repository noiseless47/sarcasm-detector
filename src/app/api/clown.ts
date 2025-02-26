// D:\clown-detector-app\pages\api\clown.ts
import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData();
        const response = await axios.post('http://127.0.0.1:5000/', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return NextResponse.json(response.data);
    } catch (error) {
        console.error('Error processing request:', error);
        return NextResponse.json({ error: 'Error processing request' }, { status: 500 });
    }
}