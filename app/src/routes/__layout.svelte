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
{status}
