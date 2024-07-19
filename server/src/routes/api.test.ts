import request from 'supertest';
import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import apiRoutes from './api';
import Message from '../models/Message';
import axios from 'axios';

// Load environment variables
dotenv.config();

// Create and configure the Express app
const app = express();
app.use(express.json());
app.use('/api', apiRoutes);

// Mock the Message model
jest.mock('../models/Message', () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      save: jest.fn().mockResolvedValue({})
    }))
  };
});

jest.mock('axios');

describe('POST /api/process-text', () => {
  beforeAll(async () => {
    await mongoose.connect(process.env.DATABASE_URL!, {});
  });

  afterAll(async () => {
    await mongoose.connection.close();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return a response with AI generated text', async () => {
    const aiResponse = { text: 'AI response' };
    (axios.post as jest.Mock).mockResolvedValue({ data: aiResponse });

    const response = await request(app)
      .post('/api/process-text')
      .send({ text: 'Hello' });

    expect(response.status).toBe(200);
    expect(response.body.text).toBe('AI response');
  });

  it('should return 400 if no text is provided', async () => {
    const response = await request(app)
      .post('/api/process-text')
      .send({});

    expect(response.status).toBe(400);
    expect(response.text).toBe('No text provided');
  });

  it('should handle errors from the AI API', async () => {
    (axios.post as jest.Mock).mockRejectedValue(new Error('AI API error'));

    const response = await request(app)
      .post('/api/process-text')
      .send({ text: 'Hello' });

    expect(response.status).toBe(500);
    expect(response.text).toBe('Error processing text');
  });
});

