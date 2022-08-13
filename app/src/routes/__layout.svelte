<script>
  import "../app.css";

  import { invoke } from "@tauri-apps/api";
  import { onDestroy } from "svelte";

  let status = "";
  const interval = setInterval(() => {
    invoke("get_status", {}).then((response) => {
      if (status !== response) {
        status = response;
      }
    });
  }, 1000);
  onDestroy(() => clearInterval(interval));
</script>

<slot />
<div
  class="fixed bottom-0 bg-gray-300 left-0 right-0 font-mono text-gray-700 text-sm px-3"
>
  Status: {status || "OK"}
</div>
