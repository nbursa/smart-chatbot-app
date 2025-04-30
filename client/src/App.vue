<template>
  <div
    class="w-full h-full overflow-hidden bg-gray-900 flex items-center justify-center"
  >
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
