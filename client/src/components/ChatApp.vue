<template>
  <div class="flex flex-col max-w-3xl mx-auto bg-gray-900 text-gray-100">
    <div class="flex-1 overflow-y-auto p-4">
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

interface Message {
  id: number;
  text: string;
}

export default defineComponent({
  name: 'ChatApp',
  components: {
    MessageList,
    MessageInput,
  },
  setup() {
    const messages = ref<Message[]>([
      { id: 1, text: 'Hello! How can I help you today?' },
    ]);

    const sendMessage = (message: string) => {
      messages.value.push({ id: messages.value.length + 1, text: message });
    };

    return {
      messages,
      sendMessage,
    };
  },
});
</script>