<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import { invoke } from "@tauri-apps/api";

  onMount(async () => {
    invoke("init", { kind: "large" }).then((response) => {
      if (response === true) {
        goto("/text");
      }
    });
  });
</script>

<h1
  class="text-4xl text-center flex flex-col min-h-screen place-content-center text-gray-600"
>
  Initializing...
</h1>
