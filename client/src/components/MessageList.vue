<template>
  <div class="space-y-4 w-full">
    <div
      v-for="message in messages"
      :key="message.id"
      class="p-4 bg-gray-800 rounded-lg"
    >
      <p class="text-xs mb-3 text-gray-400 text-left">
        {{ message.meta.sender.toUpperCase() }} -
        {{ new Date(message.meta.timestamp).toLocaleString() }}
      </p>

      <p class="text-left">
        <span v-if="message.meta.typing" class="typing-dots">
          <span>.</span><span>.</span><span>.</span>
        </span>
        <span v-else>{{ message.text }}</span>
      </p>

      <p v-if="message.meta.additionalInfo" class="text-xs text-gray-500">
        {{ message.meta.additionalInfo }}
      </p>
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

<style scoped>
.typing-dots span {
  display: inline-block;
  animation: blink 1.4s infinite;
  font-weight: bold;
  font-size: 1.5rem;
  margin-right: 2px;
}

.typing-dots span:nth-child(2) {
  animation-delay: 0.2s;
}
.typing-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  0%,
  20% {
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0;
  }
}
</style>
