<script>
  import { invoke } from "@tauri-apps/api";
  import { onMount } from "svelte";

  let name = "";
  onMount(() => {
    invoke("greet", { name: "World" }).then((response) => {
      name = response;
    });
  });
</script>

<h1>Welcome to SvelteKit</h1>
<p>
  {name}
</p>
