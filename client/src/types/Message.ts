export interface Meta {
  sender: 'user' | 'ai';
  timestamp: Date;
  additionalInfo?: string;
}

export interface Message {
  id: number;
  text: string;
  meta: Meta;
}