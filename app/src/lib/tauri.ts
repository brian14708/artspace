import type tauri from "@tauri-apps/api";

async function getApi(): Promise<typeof tauri> {
  return import("@tauri-apps/api");
}

export { getApi };
