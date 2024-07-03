import { Router, Request, Response } from 'express';
import axios from 'axios';

const router = Router();

router.post('/process-text', async (req: Request, res: Response) => {
  const { text } = req.body;

  if (!text) {
    return res.status(400).send('No text provided');
  }

  try {
    const response = await axios.post(process.env.ML_API_URL!, { text });
    res.json(response.data);
  } catch (error) {
    res.status(500).send('Error processing text');
  }
});

export default router;