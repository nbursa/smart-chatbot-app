import { Schema, model } from 'mongoose';

interface Meta {
  sender: 'user' | 'ai';
  timestamp: Date;
  additionalInfo?: string;
}

interface Message {
  id: number;
  text: string;
  meta: Meta;
}

const metaSchema = new Schema<Meta>({
  sender: { type: String, enum: ['user', 'ai'], required: true },
  timestamp: { type: Date, default: Date.now },
  additionalInfo: { type: String, required: false },
});

const messageSchema = new Schema<Message>({
  id: { type: Number, required: true },
  text: { type: String, required: true },
  meta: { type: metaSchema, required: true },
});

const MessageModel = model<Message>('Message', messageSchema);

export default MessageModel;