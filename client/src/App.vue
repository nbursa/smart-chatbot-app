<template>
  <div class="max-h-screen flex items-center justify-center bg-gray-900 text-gray-100">
    <ChatApp />
  </div>
</template>

<script lang="ts">
import ChatApp from './components/ChatApp.vue';
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'App',
  components: {
    ChatApp,
  },
});
</script>