<template>
  <div class="space-y-4">
    <div v-for="message in messages" :key="message.id" class="p-4 bg-gray-800 rounded-lg">
      {{ message.text }}
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue';

interface Message {
  id: number;
  text: string;
}

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