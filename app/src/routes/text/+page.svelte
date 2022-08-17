<script lang="ts">
  import { goto } from "$app/navigation";
  import Loading from "$lib/Loading.svelte";
  import { invoke } from "@tauri-apps/api";

  const RATIOS = [
    [1, 1 / 2],
    [1, 3 / 4],
    [1, 1],
    [3 / 4, 1],
  ];
  let text = "";
  let selected = 2;
  let loading = false;

  async function submit() {
    loading = true;
    invoke("step_text", { text: text }).then((response) => {
      if (response === true) {
        goto(`/generate?w=${RATIOS[selected][0]}&h=${RATIOS[selected][1]}`);
      }
    });
  }
</script>

{#if loading}
  <Loading />
{/if}
<div
  class="flex flex-col justify-center min-h-screen max-w-[800px] m-auto text-center"
>
  <form on:submit|preventDefault={submit}>
    <div class="p-8">
      <input
        type="text"
        bind:value={text}
        placeholder="Enter a prompt..."
        class="border-4 border-zinc-200 rounded-lg p-2 w-full focus:ring-2 ring-zinc-600"
      />
    </div>
    <div class="flex justify-around items-center mx-3">
      {#each RATIOS as [w, h], i}
        <div
          class={"bg-zinc-200 border-4 rounded-lg " +
            (selected === i
              ? "border-zinc-600"
              : "border-zinc-200 cursor-pointer")}
          style:width={`${Math.round(w * 80)}px`}
          style:height={`${Math.round(h * 80)}px`}
          on:click={() => {
            selected = i;
          }}
        />
      {/each}
    </div>
    <div class="pt-16">
      <button
        type="submit"
        class="uppercase bg-zinc-600 p-3 rounded-lg font-bold text-white"
      >
        Generate
      </button>
    </div>
  </form>
</div>
