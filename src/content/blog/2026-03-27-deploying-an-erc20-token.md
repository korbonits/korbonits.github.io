---
title: "I Deployed My First Smart Contract in an Afternoon"
date: 2026-03-27
draft: true
description: A first-principles walkthrough of building and deploying a minimal ERC-20 token to Ethereum mainnet — no frameworks, no abstractions, just Solidity, Hardhat, and Uniswap.
tags:
  - ethereum
  - solidity
  - web3
  - smart-contracts
---

I'd been meaning to learn how smart contracts actually work for a while. Not "how to use OpenZeppelin" or "how to deploy with Remix" — how they work. What's in an ERC-20. What happens when you deploy. What gas fees are paying for.

So I sat down and did it from scratch: a hand-rolled ERC-20 token, deployed to Ethereum mainnet, with a live Uniswap liquidity pool. This is that walkthrough.

## What is an ERC-20?

ERC-20 is a standard interface for fungible tokens on Ethereum. "Fungible" means every unit is identical — one VIBE is the same as any other VIBE, the same way every dollar bill is interchangeable.

The standard defines a handful of functions every compliant token must implement. That's it. There's no magic — it's just a convention that wallets and exchanges know how to talk to.

## The Contract

No OpenZeppelin. I wanted to see every line.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VibeToken {
    string public name = "VibeToken";
    string public symbol = "VIBE";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    address public owner;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalSupply = 1_000_000 * 10 ** decimals;
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(balanceOf[from] >= amount, "insufficient balance");
        require(allowance[from][msg.sender] >= amount, "insufficient allowance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }

    function mint(address to, uint256 amount) public onlyOwner {
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function burn(uint256 amount) public {
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
    }
}
```

A few things that surprised me coming from traditional software:

**Balances are just a mapping.** There's no "token" object anywhere. The entire ledger is `mapping(address => uint256) public balanceOf` — a hash map from wallet address to balance. When you "send" tokens, you're just decrementing one entry and incrementing another.

**`address(0)` is the null address.** Minting emits a `Transfer` from `address(0)` — the convention for "these tokens came from nowhere." Burning emits a `Transfer` to `address(0)` — "these tokens went nowhere." Wallets and explorers use this to display mint/burn events correctly.

**The `approve` + `transferFrom` pattern.** This is how DEXes work. You can't let a contract move your tokens without your permission, so you first `approve` an allowance, then the contract calls `transferFrom` to pull exactly that amount. Two transactions, but atomic safety.

**`modifier` is a function wrapper.** The `onlyOwner` modifier is reusable access control — `_;` means "run the function body here." Any function tagged `onlyOwner` rejects callers who aren't the deployer. Mint is dangerous (creates money from nothing), so it gets this guard.

## The Tooling

I used **Hardhat** to compile and deploy. Hardhat gives you a local EVM for testing, a task runner for deployment scripts, and a plugin ecosystem for verification.

A few friction points worth documenting:

**Hardhat 3 + Node.js 25 has a broken HTTP client.** The compiler downloader uses a fetch implementation that fails with `UNABLE_TO_GET_ISSUER_CERT_LOCALLY` on Node 25. The fix: `NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem` points Node at macOS's system cert store. Add it to your shell profile and forget about it.

**You need an RPC endpoint.** Deploying to Sepolia means talking to an Ethereum node. I used Alchemy's free tier — create an app, select Ethereum Sepolia, copy the HTTPS URL into `.env`.

**You need testnet ETH.** Gas fees on testnet are paid in fake ETH. Most faucets have gatekeeping (minimum mainnet balance, social login, etc.). The Google Cloud faucet at `cloud.google.com/application/web3/faucet/ethereum/sepolia` worked without friction — just a Google account.

## Deployment

The deployment script is minimal:

```js
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  const VibeToken = await hre.ethers.getContractFactory("VibeToken");
  const token = await VibeToken.deploy();
  await token.waitForDeployment();

  console.log("VibeToken deployed to:", await token.getAddress());
}

main().catch((err) => { console.error(err); process.exit(1); });
```

```
npx hardhat run scripts/deploy.js --network sepolia
```

Output:
```
Deploying with account: 0xf8C4795DFdaAC904CF2a380e3599a628E902C653
VibeToken deployed to: 0x2B2FD8fDdcca4247a1f0370c0F26Dae91648d2F1
```

That contract is now permanent. Anyone can call it. The 1,000,000 VIBE tokens exist in my wallet. The whole thing cost roughly 0.001 Sepolia ETH in gas.

## Testing

Tests run locally against Hardhat's in-process EVM — no testnet, no waiting, no gas.

```js
it("owner can mint new tokens", async function () {
  await token.mint(alice.address, ethers.parseUnits("500", 18));
  expect(await token.balanceOf(alice.address)).to.equal(ethers.parseUnits("500", 18));
  expect(await token.totalSupply()).to.equal(ethers.parseUnits("1000500", 18));
});

it("non-owner cannot mint", async function () {
  await expect(
    token.connect(alice).mint(alice.address, 1)
  ).to.be.revertedWith("not owner");
});
```

`getSigners()` gives you fake funded accounts. `token.connect(alice)` sends transactions as Alice. `revertedWith` asserts the require message. Nine tests, under a second.

The loop is: write contract → test locally → deploy to testnet when confident. The testnet deployment is the last step, not the first.

## Listing on Uniswap

Deploying a token is one thing. Making it tradeable is another.

Uniswap is a decentralized exchange — no company, no order book, no account. It's a set of smart contracts anyone can interact with directly. The core primitive is the **liquidity pool**: a contract holding two tokens that lets anyone swap between them. The price adjusts automatically based on a constant product formula (`x * y = k`). When you buy VIBE, you're adding ETH and removing VIBE from the pool, which shifts the ratio and raises the price.

To create a VIBE/ETH pool, I wrote a script that does four things:

**1. Wrap ETH → WETH.** Uniswap only works with ERC-20 tokens. Raw ETH isn't an ERC-20, so you wrap it first. WETH is just ETH with a `deposit()`/`withdraw()` interface — 1 WETH is always redeemable for 1 ETH.

**2. Approve the position manager.** Uniswap's contracts need permission to pull tokens from your wallet. This is the `approve` + `transferFrom` pattern from the contract — it applies here too. You approve the Uniswap NonfungiblePositionManager to spend your WETH and VIBE.

**3. Create and initialize the pool.** The pool doesn't exist until someone creates it. You pick a fee tier (0.3% is standard) and set the **starting price** — expressed as `sqrtPriceX96`, which is `sqrt(token1/token0) * 2^96`. I set it so 1 ETH = 1,000,000 VIBE, which is arbitrary on testnet but determines where trading starts.

**4. Mint a liquidity position.** You deposit both tokens into the pool and receive an NFT representing your share. Uniswap v3 uses "concentrated liquidity" — you can choose a price range to earn fees in. I used full range (equivalent to Uniswap v2 behavior) to keep it simple.

```js
const tx = await positionManager.mint({
  token0,
  token1,
  fee: 3000,
  tickLower: -887220,  // full range
  tickUpper: 887220,
  amount0Desired: amount0,
  amount1Desired: amount1,
  amount0Min: 0,
  amount1Min: 0,
  recipient: signer.address,
  deadline: Math.floor(Date.now() / 1000) + 60 * 20,
});
```

The pool went live at transaction `0xed4184b73e9848ede46622b4c2afdf392220c94a6b8509ab31ebd400af75ef44`.

## The First Swap

Once the pool existed, I went to app.uniswap.org, switched MetaMask to Sepolia, enabled testnet mode in the Uniswap settings, and pasted the VIBE contract address into the token selector.

Swapped 0.0001 ETH for VIBE. It worked.

What happened under the hood: Uniswap's router called `approve` on my behalf, then called `transferFrom` on the VIBE contract to move tokens out of the pool and into my wallet. The same two functions I wrote by hand earlier in the day.

The price moved slightly after the swap — the pool now has slightly less VIBE and slightly more ETH, so the ratio shifted. That's the AMM doing its job.

## Going to Mainnet

Once everything worked on Sepolia, mainnet was anticlimactic — exactly as it should be.

Added a mainnet network to `hardhat.config.js`, pointed it at an Alchemy mainnet RPC, and ran the same deploy script:

```
npx hardhat run scripts/deploy.js --network mainnet
```

Gas was 0.165 Gwei. The whole deployment cost $0.40.

```
VibeToken deployed to: 0x4d5007d5717795331e8b21b3cd584f7bfe505926
```

Verified on Etherscan:
```
NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem npx hardhat verify --network mainnet 0x4d5007d5717795331e8b21b3cd584f7bfe505926
```

Then ran the same liquidity script against mainnet — seeded the pool with 0.01 ETH and 10,000 VIBE. VIBE is now tradeable by anyone on Uniswap mainnet.

Contract: [0x4d5007...505926](https://etherscan.io/address/0x4d5007d5717795331e8b21b3cd584f7bfe505926#code)

The testnet-first workflow paid off. By the time I touched mainnet, every step had already run successfully. The only surprise was a Hardhat bug where the ethers.js transaction formatter choked on a contract deployment response (the `to` field is empty for contract creation transactions). The transaction went through fine — the error was in the post-deploy parsing, not the deployment itself.

## What I Actually Learned

The mental model shift: **a smart contract is a program that lives at an address on a shared computer.** The EVM is that computer. Every Ethereum node runs it. When you deploy, you're uploading bytecode to a permanent address. When someone calls a function, every node executes it and agrees on the result.

Tokens aren't stored anywhere special. There's no token registry. The "balance" is just a number in a mapping inside a contract. MetaMask knows you own VIBE because it calls `balanceOf(yourAddress)` on the contract and displays the result.

Gas is payment for computation. Every opcode costs gas. The deployment transaction includes the contract bytecode, which costs more gas than a simple transfer because there's more data. Miners/validators won't run your code for free.

The `owner` pattern I implemented is primitive — a real contract would use something like OpenZeppelin's `Ownable` with ownership transfer and renouncement. But I now understand *why* that pattern exists, which I wouldn't if I'd started with the library.

The code is at [github.com/korbonits/vibe-token](https://github.com/korbonits/vibe-token).
