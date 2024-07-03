<template>
  <div class="space-y-4">
    <div v-for="message in messages" :key="message.id" class="p-4 bg-gray-800 rounded-lg">
      <p class="text-sm text-gray-400">{{ message.meta.sender }} - {{ new Date(message.meta.timestamp).toLocaleString() }}</p>
      <p>{{ message.text }}</p>
      <p v-if="message.meta.additionalInfo" class="text-xs text-gray-500">{{ message.meta.additionalInfo }}</p>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue';
import { Message } from '../types/Message';

export default defineComponent({
  name: 'MessageList',
  props: {
    messages: {
      type: Array as PropType<Message[]>,
      required: true,
    },
  },
});
</script>