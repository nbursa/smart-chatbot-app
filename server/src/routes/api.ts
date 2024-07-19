import { Router, Request, Response } from 'express';
import axios, { AxiosError } from 'axios';
import Message from '../models/Message';

const router = Router();

router.post('/process-text', async (req: Request, res: Response) => {
  const { text } = req.body;

  if (!text) {
    console.error('No text provided');
    return res.status(400).send('No text provided');
  }

  try {
    const userMessage = new Message({
      id: Date.now(),
      text: text,
      meta: {
        sender: 'user',
        timestamp: new Date(),
      },
    });
    await userMessage.save();

    const aiResponse = await axios.post<{ text: string }>(process.env.ML_API_URL!, { text });

    const newAIMessage = new Message({
      id: Date.now() + 1,
      text: aiResponse.data.text,
      meta: {
        sender: 'ai',
        timestamp: new Date(),
      },
    });
    await newAIMessage.save();

    res.json(newAIMessage);
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response) {
        console.error('Server responded with non-success status:', axiosError.response.data);
      } else if (axiosError.request) {
        console.error('No response received:', axiosError.request);
      } else {
        console.error('Error setting up the request:', axiosError.message);
      }
    } else {
      console.error('Unexpected error:', error);
    }
    res.status(500).send('Error processing text');
  }
});

export default router;
