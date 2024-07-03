<template>
  <div class="flex items-center space-x-4">
    <input
      v-model="message"
      @keyup.enter="submitMessage"
      type="text"
      placeholder="Type a message..."
      class="flex-1 p-2 border border-gray-600 rounded-lg bg-gray-700 text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
    />
    <button @click="submitMessage" class="p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500">
      Send
    </button>
  </div>
</template>

<script lang="ts">
import { ref, defineComponent } from 'vue';

export default defineComponent({
  name: 'MessageInput',
  emits: ['sendMessage'],
  setup(_, { emit }) {
    const message = ref<string>('');

    const submitMessage = () => {
      if (message.value.trim()) {
        emit('sendMessage', message.value);
        message.value = '';
      }
    };

    return {
      message,
      submitMessage,
    };
  },
});
</script>