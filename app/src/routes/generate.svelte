<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/stores";
  import { save } from "@tauri-apps/api/dialog";
  import { invoke } from "@tauri-apps/api";
  import Loading from "$lib/Loading.svelte";

  let images: Array<string | null> = [null];
  let processing: number | null = null;
  let loading = false;

  const savef = async function (i: number) {
    loading = true;
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
    loading = false;
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

{#if loading}
  <Loading />
{/if}
<div
  class="flex flex-col justify-center min-h-screen max-w-[800px] m-auto text-center"
>
  <div class="flex flex-row flex-wrap justify-center gap-4 m-4">
    {#each images as img, i}
      <div
        class="bg-no-repeat bg-contain bg-center min-w-[300px] min-h-[300px] relative"
        style:background-image={img ? `url(${img})` : "none"}
      >
        <div
          class={"absolute top-0 right-0 bottom-0 left-0 flex items-center justify-center " +
            (img
              ? "opacity-0 bg-none hover:bg-white hover:bg-opacity-50 transition-all hover:opacity-100 flex-col gap-2"
              : "")}
        >
          {#if img}
            <button
              class="uppercase bg-zinc-600 p-3 rounded-lg font-bold text-white text-sm py-2"
              on:click={() => {
                images[i] = null;
                check();
              }}
            >
              Delete
            </button>
            {#if processing === null}
              <button
                class="uppercase bg-zinc-600 p-3 rounded-lg font-bold text-white text-sm py-2"
                on:click={() => {
                  savef(i);
                }}>Save</button
              >
            {/if}
          {:else}
            Loading...
          {/if}
        </div>
      </div>
    {/each}
  </div>

  <div class="p-4 pb-6">
    <a href="/text">
      <button
        class="uppercase bg-zinc-600 p-3 rounded-lg font-bold text-white text-sm py-2"
      >
        Back
      </button>
    </a>

    <button
      class="uppercase bg-zinc-600 p-3 rounded-lg font-bold text-white text-sm py-2"
      on:click={() => {
        images.push(null);
        images = images;
        check();
      }}
    >
      Add
    </button>
  </div>
</div>
