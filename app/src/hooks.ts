import type { Handle } from "@sveltejs/kit";

const handle: Handle = async function ({ event, resolve }) {
  return resolve(event, { ssr: false });
};
export { handle };
