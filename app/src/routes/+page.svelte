<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import { getApi } from "$lib/tauri";

  onMount(async () => {
    const { invoke } = await getApi();

    invoke("init", { kind: "small" }).then((response) => {
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
