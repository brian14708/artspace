<script>
  import { invoke } from "@tauri-apps/api";
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";

  onMount(() => {
    invoke("init", { kind: "small" }).then((response) => {
      if (response === true) {
        goto("/text");
      }
    });
  });
</script>

<h1>Loading...</h1>
