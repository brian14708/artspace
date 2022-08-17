<script lang="ts">
  import "../app.css";

  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api";

  let status = "";
  let interval: ReturnType<typeof setInterval> | null = null;
  onMount(async () => {
    interval = setInterval(() => {
      invoke("get_status", {}).then((response) => {
        if (status !== response) {
          status = response as string;
        }
      });
    }, 1000);
  });

  onDestroy(() => interval && clearInterval(interval));
</script>

<slot />
<div
  class="fixed bottom-0 bg-gray-300 left-0 right-0 font-mono text-gray-700 text-sm px-3"
>
  Status: {status || "OK"}
</div>
