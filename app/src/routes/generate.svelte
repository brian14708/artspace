<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/stores";
  import { save } from "@tauri-apps/api/dialog";

  let images: Array<string | null> = [null];
  let processing: number | null = null;

  import { invoke } from "@tauri-apps/api";
  const savef = async function (i: number) {
    let filePath = await save({
      filters: [
        {
          name: "Image",
          extensions: ["png"],
        },
      ],
    });
    if (!filePath) {
      return;
    }
    if (!filePath.endsWith(".png")) {
      filePath += ".png";
    }
    await invoke("step_post", { idx: i, path: filePath });
  };

  const w = parseFloat($page.url.searchParams.get("w") || "1");
  const h = parseFloat($page.url.searchParams.get("h") || "1");
  function check() {
    if (processing === null) {
      for (let i = 0; i < images.length; i++) {
        if (images[i] === null) {
          processing = i;

          invoke("step_diffuse", { w: w, h: h, idx: i }).then((response) => {
            if (response) {
              let arr = Uint8Array.from(response as Array<number>);
              let b = new Blob([arr]);
              let url = URL.createObjectURL(b);
              images[i] = url;
              processing = null;
              check();
            }
          });
          return;
        }
      }
    }
  }

  onMount(check);
</script>

<div class="grid grid-cols-2 grid-flow-row text-center">
  {#each images as img, i}
    {#if img}
      <div
        class="bg-no-repeat bg-contain bg-center min-w-[300px] min-h-[300px]"
        style:background-image={`url(${img})`}
      >
        <button
          on:click={() => {
            savef(i);
          }}>OK</button
        >
        <button
          on:click={() => {
            images[i] = null;
            check();
          }}>X</button
        >
      </div>
    {:else}
      <div class="min-w-[300px] min-h-[300px]" />
    {/if}
  {/each}
</div>
<a href="/text">Back</a>
