<template>
  <div
    class="flex flex-col w-full max-w-xl max-h-[90vh] bg-gray-900 text-gray-100 border border-gray-700 rounded-lg overflow-hidden"
  >
    <h1 class="text-4xl mt-4 mb-8 text-center">The Roasted Oak Barista</h1>
    <div
      ref="chatContainer"
      class="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col-reverse gap-4"
    >
      <MessageList :messages="messages" />
    </div>
    <div class="p-4 border-t border-gray-700">
      <MessageInput @sendMessage="sendMessage" :disabled="isLoading" />
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
    const isLoading = ref(false);

    const messages = ref<Message[]>([
      {
        id: 1,
        text: 'Welcome to The Roasted Oak ☕ How can I assist you today? I can recommend drinks, explain our beans, or help with your order.',
        meta: { sender: 'ai', typing: false, timestamp: new Date() },
      },
    ]);

    const sendMessage = async (message: string) => {
      try {
        const newMessage: Message = {
          id: messages.value.length + 1,
          text: message,
          meta: { sender: 'user', typing: false, timestamp: new Date() },
        };
        messages.value.push(newMessage);

        isLoading.value = true;

        // Temporary loader
        const loadingMessage: Message = {
          id: newMessage.id + 1,
          text: '...',
          meta: { sender: 'ai', typing: true, timestamp: new Date() },
        };
        messages.value.push(loadingMessage);

        const response = await axios.post<Message>(
          `${import.meta.env.VITE_API_URL}/process-text`,
          { text: message }
        );

        messages.value.pop(); // Remove "..." loader

        if (response.data) {
          const aiResponse: Message = response.data;
          messages.value.push(aiResponse);
        } else {
          console.error('Invalid response format from server:', response);
        }
      } catch (error) {
        console.error('Error sending message:', error);
        messages.value.pop(); // Remove loader on failure
      } finally {
        isLoading.value = false;
      }
    };

    return {
      messages,
      sendMessage,
      isLoading,
    };
  },
});
</script>
