<template>
  <div
    class="flex flex-col w-full max-w-xl max-h-[90vh] bg-gray-900 text-gray-100 border border-gray-700 rounded-lg overflow-hidden"
  >
    <div
      ref="chatContainer"
      class="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col-reverse gap-4"
    >
      <MessageList :messages="messages" />
    </div>
    <div class="p-4 border-t border-gray-700">
      <MessageInput @sendMessage="sendMessage" />
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import MessageList from './MessageList.vue';
import MessageInput from './MessageInput.vue';
import axios from 'axios';
import { Message } from '../types';

export default defineComponent({
  name: 'ChatApp',
  components: {
    MessageList,
    MessageInput,
  },
  setup() {
    const messages = ref<Message[]>([
      {
        id: 1,
        text: 'Hello! How can I help you today?',
        meta: { sender: 'ai', timestamp: new Date() },
      },
    ]);

    const sendMessage = async (message: string) => {
      try {
        const newMessage: Message = {
          id: messages.value.length + 1,
          text: message,
          meta: { sender: 'user', timestamp: new Date() },
        };
        messages.value.push(newMessage);

        const response = await axios.post<Message>(
          `${import.meta.env.VITE_API_URL}/process-text`,
          { text: message }
        );

        if (response.data) {
          const aiResponse: Message = response.data;
          messages.value.push(aiResponse);
        } else {
          console.error('Invalid response format from server:', response);
        }
      } catch (error) {
        console.error('Error sending message:', error);
      }
    };

    return {
      messages,
      sendMessage,
    };
  },
});
</script>
