import express, { Request, Response } from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import axios from 'axios';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());

app.get('/', (req: Request, res: Response) => {
    res.send('Server is running');
});

app.post('/process-text', async (req: Request, res: Response) => {
    const { text } = req.body;

    if (!text) {
        return res.status(400).send('No text provided');
    }

    try {
        const response = await axios.post('http://localhost:8000/process', { text });
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Error processing text');
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});