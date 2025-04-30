export interface Meta {
  sender: 'user' | 'ai';
  typing: boolean;
  timestamp: Date;
  additionalInfo?: string;
}

export interface Message {
  id: number;
  text: string;
  meta: Meta;
}
